"""Fail-closed parsing and exact joins for aggregate K=4 winner summaries."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recovar.em.global_winner_summary import MAX_SUPPORTED_BYTES, SCHEMA, sha256_file


@dataclass(frozen=True)
class WinnerSummary:
    label: str
    engine: str
    identity: np.ndarray
    winner: np.ndarray
    runner_up: np.ndarray
    margin: np.ndarray
    class_scores: np.ndarray
    class_pose_indices: np.ndarray
    class_second_scores: np.ndarray
    class_second_pose_indices: np.ndarray
    class_within_pose_margin: np.ndarray
    class_posterior_mass: np.ndarray
    significant_count: np.ndarray
    class_log_evidence: np.ndarray | None
    global_log_z: np.ndarray | None
    pose_topology: tuple[int, int]
    metadata: dict
    artifact_paths: tuple[Path, ...]


def _as_scalar(array: np.ndarray, name: str):
    if array.shape != ():
        raise ValueError(f"{name} must be scalar, got shape {array.shape}")
    return array.item()


def load_recovar_summary(path: str | Path, *, label: str) -> WinnerSummary:
    path = Path(path).resolve()
    if path.stat().st_size > MAX_SUPPORTED_BYTES:
        raise ValueError(f"RECOVAR artifact exceeds {MAX_SUPPORTED_BYTES} bytes: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "schema",
            "metadata_json",
            "original_index_zero_based",
            "class_best_raw_score_pre_prior",
            "class_best_total_score",
            "class_best_absolute_log_score_with_image_offset",
            "class_best_pose_index",
            "class_second_best_raw_score_pre_prior",
            "class_second_best_total_score",
            "class_second_best_absolute_log_score_with_image_offset",
            "class_second_best_pose_index",
            "class_within_pose_margin",
            "winner_class_zero_based",
            "runner_up_class_zero_based",
            "winner_margin",
            "class_posterior_mass",
            "class_log_evidence",
            "global_log_z",
            "significant_count",
            "n_rotations",
            "n_translations",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"RECOVAR artifact lacks required fields: {sorted(missing)}")
        schema = str(_as_scalar(payload["schema"], "schema"))
        if schema != SCHEMA:
            raise ValueError(f"unsupported RECOVAR schema {schema!r}")
        metadata = json.loads(str(_as_scalar(payload["metadata_json"], "metadata_json")))
        identity = np.asarray(payload["original_index_zero_based"], dtype=np.int64)
        raw_scores_native = payload["class_best_raw_score_pre_prior"]
        total_scores_native = payload["class_best_total_score"]
        if raw_scores_native.dtype != np.float32 or total_scores_native.dtype != np.float32:
            raise ValueError("RECOVAR native class scores must be float32")
        if not np.array_equal(raw_scores_native, total_scores_native):
            raise ValueError("RECOVAR firstiter_cc raw and total scores must be identical")
        class_scores = np.asarray(raw_scores_native, dtype=np.float32)
        absolute_scores_native = payload["class_best_absolute_log_score_with_image_offset"]
        absolute_second_scores_native = payload["class_second_best_absolute_log_score_with_image_offset"]
        if absolute_scores_native.dtype != np.float32 or absolute_second_scores_native.dtype != np.float32:
            raise ValueError("RECOVAR absolute diagnostic class scores must be float32")
        if absolute_scores_native.shape != class_scores.shape or absolute_second_scores_native.shape != class_scores.shape:
            raise ValueError("RECOVAR absolute diagnostic class scores have unexpected shapes")
        if not np.all(np.isfinite(absolute_scores_native)) or not np.all(np.isfinite(absolute_second_scores_native)):
            raise ValueError("RECOVAR absolute diagnostic class scores contain non-finite values")
        class_pose = np.asarray(payload["class_best_pose_index"], dtype=np.int32)
        class_second_scores_native = payload["class_second_best_raw_score_pre_prior"]
        class_second_total_scores_native = payload["class_second_best_total_score"]
        if class_second_scores_native.dtype != np.float32 or class_second_total_scores_native.dtype != np.float32:
            raise ValueError("RECOVAR native second-best class scores must be float32")
        if not np.array_equal(class_second_scores_native, class_second_total_scores_native):
            raise ValueError("RECOVAR firstiter_cc second-best raw and total scores must be identical")
        class_second_scores = np.asarray(class_second_scores_native, dtype=np.float32)
        class_second_pose = np.asarray(payload["class_second_best_pose_index"], dtype=np.int32)
        class_within_pose_margin = np.asarray(payload["class_within_pose_margin"], dtype=np.float32)
        winner = np.asarray(payload["winner_class_zero_based"], dtype=np.int32)
        runner_up = np.asarray(payload["runner_up_class_zero_based"], dtype=np.int32)
        margin = np.asarray(payload["winner_margin"], dtype=np.float32)
        class_posterior_mass = np.asarray(payload["class_posterior_mass"], dtype=np.float32)
        class_log_evidence = np.asarray(payload["class_log_evidence"], dtype=np.float64)
        global_log_z = np.asarray(payload["global_log_z"], dtype=np.float64)
        significant_count = np.asarray(payload["significant_count"], dtype=np.int32)
        n_rotations = int(_as_scalar(payload["n_rotations"], "n_rotations"))
        n_translations = int(_as_scalar(payload["n_translations"], "n_translations"))
    if n_rotations <= 0 or n_translations <= 0:
        raise ValueError("RECOVAR pose topology must be positive")
    _validate_summary_arrays(
        identity,
        winner,
        runner_up,
        margin,
        class_scores,
        class_pose,
        higher_is_better=True,
        allow_tied_selected_winner=True,
    )
    _validate_within_class_runner_up(
        class_scores,
        class_pose,
        class_second_scores,
        class_second_pose,
        class_within_pose_margin,
        higher_is_better=True,
    )
    if metadata.get("schema") != SCHEMA or metadata.get("engine") != "recovar":
        raise ValueError("RECOVAR metadata schema/engine mismatch")
    if int(metadata.get("expected_particles", -1)) != identity.size:
        raise ValueError("RECOVAR metadata particle count mismatch")
    if int(metadata.get("expected_classes", -1)) != 4:
        raise ValueError("RECOVAR metadata class count mismatch")
    if int(metadata.get("score_element_bytes", -1)) != 4:
        raise ValueError("RECOVAR metadata does not describe native float32 scores")
    if path.stat().st_size > int(metadata.get("max_bytes", -1)):
        raise ValueError("RECOVAR artifact exceeds its declared byte cap")
    for provenance_key in (
        "run_id",
        "source_id",
        "executable_sha256",
        "gpu_uuid",
        "input_manifest_sha256",
        "dispatch_oracle_sha256",
    ):
        if not metadata.get(provenance_key):
            raise ValueError(f"RECOVAR metadata lacks {provenance_key}")
    expected_recovar_semantics = {
        "score_mode": "firstiter_cc_offset_free_normalized_log_score_higher_is_better",
        "raw_score_semantics": (
            "per-class best native float32 normalized-CC log score before class/orientation/translation priors "
            "and before the class-common per-image normalization offset"
        ),
        "total_score_semantics": ("identical to raw score in firstiter_cc because RELION bypasses priors before WTA"),
        "absolute_score_semantics": (
            "diagnostic float32 normalized-CC log score after adding the class-common per-image normalization "
            "offset; retained for absolute-score/evidence context and not used for class or pose margins"
        ),
        "evidence_semantics": "per-class logsumexp evidence before firstiter_cc one-hot posterior",
        "posterior_semantics": "post-firstiter_cc one-hot class mass",
        "winner_semantics": "actual joint class-pose argmax of native offset-free float32 scores before WTA",
        "support_semantics": "post-firstiter_cc exactly one global class-pose sample per particle",
        "pre_wta_support_semantics": "all coarse candidates scored; no posterior threshold before WTA",
        "significant_count_semantics": "post-WTA global class-pose support cardinality; exactly one",
        "within_class_runner_up_semantics": (
            "second-highest distinct class-local coarse pose score before priors and WTA"
        ),
    }
    for semantics_key, expected_value in expected_recovar_semantics.items():
        if metadata.get(semantics_key) != expected_value:
            raise ValueError(f"RECOVAR metadata has unexpected {semantics_key}")
    if class_posterior_mass.shape != class_scores.shape or not np.array_equal(
        class_posterior_mass.sum(axis=1), np.ones(identity.size, dtype=np.float32)
    ):
        raise ValueError("RECOVAR class posterior mass must be normalized per particle")
    if class_log_evidence.shape != class_scores.shape or global_log_z.shape != (identity.size,):
        raise ValueError("RECOVAR evidence/log-Z arrays have unexpected shapes")
    if not np.all(np.isfinite(class_log_evidence)) or not np.all(np.isfinite(global_log_z)):
        raise ValueError("RECOVAR evidence/log-Z arrays contain non-finite values")
    if significant_count.shape != (identity.size,) or not np.all(significant_count == 1):
        raise ValueError("RECOVAR firstiter_cc significant count must be exactly one per particle")
    return WinnerSummary(
        label=label,
        engine="recovar",
        identity=identity,
        winner=winner,
        runner_up=runner_up,
        margin=margin,
        class_scores=class_scores,
        class_pose_indices=class_pose,
        class_second_scores=class_second_scores,
        class_second_pose_indices=class_second_pose,
        class_within_pose_margin=class_within_pose_margin,
        class_posterior_mass=class_posterior_mass,
        significant_count=significant_count,
        class_log_evidence=class_log_evidence,
        global_log_z=global_log_z,
        pose_topology=(n_rotations, n_translations),
        metadata=metadata,
        artifact_paths=(path,),
    )


def _parse_comment_metadata(path: Path) -> tuple[dict[str, str], list[str]]:
    metadata: dict[str, str] = {}
    header: list[str] | None = None
    with path.open(newline="") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("# "):
                key, separator, value = line[2:].partition("=")
                if not separator or key in metadata:
                    raise ValueError(f"invalid/duplicate metadata line in {path}: {line!r}")
                metadata[key] = value
            elif line:
                header = line.split("\t")
                break
    if header is None:
        raise ValueError(f"RELION shard has no column header: {path}")
    return metadata, header


def load_relion_summary(
    directory: str | Path,
    *,
    data_star: str | Path,
    input_manifest: str | Path,
    executable: str | Path,
    dispatch_log: str | Path,
    dispatch_schedule: str | Path,
    label: str,
) -> WinnerSummary:
    directory = Path(directory).resolve()
    shards = tuple(sorted(directory.glob("rank*.tsv")))
    if not shards:
        raise ValueError(f"no RELION rank shards in {directory}")
    total_bytes = sum(path.stat().st_size for path in shards)
    if total_bytes > MAX_SUPPORTED_BYTES:
        raise ValueError(f"RELION shards exceed {MAX_SUPPORTED_BYTES} bytes")
    shared_metadata: dict[str, str] | None = None
    records: list[dict[str, str]] = []
    for path in shards:
        metadata, header = _parse_comment_metadata(path)
        if shared_metadata is None:
            shared_metadata = metadata
        elif metadata != shared_metadata:
            raise ValueError("RELION rank shards have mixed metadata/provenance")
        with path.open(newline="") as handle:
            reader = csv.DictReader((line for line in handle if not line.startswith("# ")), delimiter="\t")
            if reader.fieldnames != header:
                raise ValueError(f"RELION shard header changed while reading {path}")
            records.extend(reader)
    assert shared_metadata is not None
    if shared_metadata.get("schema") != SCHEMA:
        raise ValueError(f"unsupported RELION schema {shared_metadata.get('schema')!r}")
    expected_particles = int(shared_metadata.get("expected_particles", -1))
    if expected_particles <= 0 or len(records) != expected_particles:
        raise ValueError(f"RELION records have {len(records)} rows, expected {expected_particles}")
    if int(shared_metadata.get("expected_classes", -1)) != 4:
        raise ValueError("RELION metadata is not K=4")
    if total_bytes > int(shared_metadata.get("max_bytes", -1)):
        raise ValueError("RELION records exceed their declared byte cap")
    data_star_sha256 = sha256_file(data_star)
    input_manifest_sha256 = sha256_file(input_manifest)
    executable_sha256 = sha256_file(executable)
    dispatch_log_sha256 = sha256_file(dispatch_log)
    dispatch_schedule_sha256 = sha256_file(dispatch_schedule)
    if shared_metadata.get("input_manifest_sha256") != input_manifest_sha256:
        raise ValueError("RELION input manifest SHA-256 does not match the declared fixture manifest")
    if shared_metadata.get("executable_sha256") != executable_sha256:
        raise ValueError("RELION executable SHA-256 does not match the invoked binary")
    if shared_metadata.get("dispatch_oracle_sha256") != dispatch_schedule_sha256:
        raise ValueError("RELION dispatch manifest SHA-256 does not match the sealed schedule")

    part_id = np.asarray([int(row["part_id_zero_based"]) for row in records], dtype=np.int64)
    rank = np.asarray([int(row["mpi_rank"]) for row in records], dtype=np.int32)
    schema_version = np.asarray([int(row["schema_version"]) for row in records], dtype=np.int32)
    row_iteration = np.asarray([int(row["iteration"]) for row in records], dtype=np.int32)
    score_element_bytes = np.asarray([int(row["score_element_bytes"]) for row in records], dtype=np.int32)
    class_min = np.asarray([int(row["class_min_zero_based"]) for row in records], dtype=np.int32)
    class_max = np.asarray([int(row["class_max_zero_based"]) for row in records], dtype=np.int32)
    winner = np.asarray([int(row["winner_class_zero_based"]) for row in records], dtype=np.int32)
    runner_up = np.asarray([int(row["runner_up_class_zero_based"]) for row in records], dtype=np.int32)
    margin = np.asarray([float(row["winner_margin"]) for row in records], dtype=np.float32)
    class_scores = np.asarray(
        [[float(row[f"class{class_index}_best_raw_diff2_pre_prior"]) for class_index in range(4)] for row in records],
        dtype=np.float32,
    )
    class_total_scores = np.asarray(
        [[float(row[f"class{class_index}_best_total_diff2"]) for class_index in range(4)] for row in records],
        dtype=np.float32,
    )
    class_posterior_mass = np.asarray(
        [[float(row[f"class{class_index}_posterior_mass"]) for class_index in range(4)] for row in records],
        dtype=np.float32,
    )
    class_pose = np.asarray(
        [[int(row[f"class{class_index}_best_flat"]) for class_index in range(4)] for row in records],
        dtype=np.int64,
    )
    class_second_scores = np.asarray(
        [[float(row[f"class{class_index}_second_raw_diff2_pre_prior"]) for class_index in range(4)] for row in records],
        dtype=np.float32,
    )
    class_second_total_scores = np.asarray(
        [[float(row[f"class{class_index}_second_total_diff2"]) for class_index in range(4)] for row in records],
        dtype=np.float32,
    )
    class_second_pose = np.asarray(
        [[int(row[f"class{class_index}_second_flat"]) for class_index in range(4)] for row in records],
        dtype=np.int64,
    )
    class_within_pose_margin = np.asarray(
        [[float(row[f"class{class_index}_within_pose_margin"]) for class_index in range(4)] for row in records],
        dtype=np.float32,
    )
    significant_count = np.asarray([int(row["significant_count"]) for row in records], dtype=np.int32)
    nr_dir = np.asarray([int(row["nr_dir"]) for row in records], dtype=np.int64)
    nr_psi = np.asarray([int(row["nr_psi"]) for row in records], dtype=np.int64)
    nr_trans = np.asarray([int(row["nr_trans"]) for row in records], dtype=np.int64)
    winner_flat = np.asarray([int(row["winner_flat_index"]) for row in records], dtype=np.int64)
    winner_score = np.asarray([float(row["winner_score"]) for row in records], dtype=np.float32)
    if not np.all(schema_version == 1):
        raise ValueError("RELION rows contain an unexpected schema version")
    declared_iteration = int(shared_metadata.get("iteration", row_iteration[0]))
    if not np.all(row_iteration == declared_iteration):
        raise ValueError("RELION rows contain an unexpected/mixed iteration")
    if not np.all(score_element_bytes == 4):
        raise ValueError("RELION native score element bytes must be 4")
    if not (np.all(class_min == 0) and np.all(class_max == 3)):
        raise ValueError("RELION rows contain an unexpected class range")
    if not np.array_equal(class_scores, class_total_scores):
        raise ValueError("RELION firstiter_cc raw and total scores must be identical")
    if not np.array_equal(class_second_scores, class_second_total_scores):
        raise ValueError("RELION firstiter_cc second-best raw and total scores must be identical")
    expected_relion_semantics = {
        "score_mode": "firstiter_cc_raw_diff2_lower_is_better",
        "raw_score_semantics": "per-class minimum native diff2 before priors and WTA",
        "total_score_semantics": "identical to raw diff2 because firstiter_cc bypasses priors",
        "posterior_semantics": "post-firstiter_cc one-hot class mass",
        "winner_semantics": "actual device getArgMin of native float32 joint class-pose scores before WTA",
        "support_semantics": "post-firstiter_cc exactly one global class-pose sample per particle",
        "pre_wta_support_semantics": "all coarse candidates scored; no posterior threshold before WTA",
        "significant_count_semantics": "post-WTA global class-pose support cardinality; exactly one",
        "within_class_runner_up_semantics": (
            "second-lowest distinct class-local coarse pose diff2 before priors and WTA"
        ),
    }
    for semantics_key, expected_value in expected_relion_semantics.items():
        if shared_metadata.get(semantics_key) != expected_value:
            raise ValueError(f"RELION metadata has unexpected {semantics_key}")
    if shared_metadata.get("evidence_availability") != "unavailable":
        raise ValueError("RELION firstiter_cc evidence availability must be explicitly unavailable")
    expected_evidence_reason = "firstiter_cc bypasses exponentiation and logsumexp evidence"
    if shared_metadata.get("evidence_unavailable_reason") != expected_evidence_reason:
        raise ValueError("RELION firstiter_cc evidence-unavailable reason is unexpected")
    expected_posterior = np.zeros((expected_particles, 4), dtype=np.float32)
    expected_posterior[np.arange(expected_particles), winner] = 1.0
    if not np.array_equal(class_posterior_mass, expected_posterior):
        raise ValueError("RELION post-WTA class posterior mass is not exactly one-hot")
    if significant_count.shape != (expected_particles,) or not np.all(significant_count == 1):
        raise ValueError("RELION firstiter_cc significant count must be exactly one per particle")
    if np.unique(part_id).size != expected_particles:
        raise ValueError("RELION summary part IDs are not unique")
    if not (np.all(nr_dir == nr_dir[0]) and np.all(nr_psi == nr_psi[0]) and np.all(nr_trans == nr_trans[0])):
        raise ValueError("RELION summary has mixed score topology")
    class_stride = int(nr_dir[0] * nr_psi[0] * nr_trans[0])
    if not np.array_equal((winner_flat // class_stride).astype(np.int32), winner):
        raise ValueError("RELION selected device argmin index disagrees with winner class")
    if not np.array_equal(class_scores[np.arange(expected_particles), winner], winner_score):
        raise ValueError("RELION selected device argmin score disagrees with winner-class minimum")
    class_pose = (class_pose % class_stride).astype(np.int32)
    class_second_pose = (class_second_pose % class_stride).astype(np.int32)
    _validate_within_class_runner_up(
        class_scores,
        class_pose,
        class_second_scores,
        class_second_pose,
        class_within_pose_margin,
        higher_is_better=False,
    )

    star_identity, star_class = read_relion_identity_classes(data_star)
    star_part_ids = {part for part, _original, _class_index in star_identity.values()}
    if star_part_ids != set(part_id.tolist()):
        raise ValueError("RELION summary and data STAR do not contain the same internal identities")
    identity_by_part = {part: original for part, original, _ in star_identity.values()}
    original_identity = np.asarray([identity_by_part[int(value)] for value in part_id], dtype=np.int64)
    star_class_by_part = {part: class_index for part, _, class_index in star_identity.values()}
    expected_winner = np.asarray([star_class_by_part[int(value)] for value in part_id], dtype=np.int32)
    if not np.array_equal(expected_winner, winner):
        raise ValueError("RELION summary winners disagree with run_it001_data.star")
    dispatch_records = read_dispatch_records(dispatch_log, iteration=int(row_iteration[0]))
    schedule_records = read_dispatch_schedule_records(dispatch_schedule, iteration=int(row_iteration[0]))
    if dispatch_records.keys() != schedule_records.keys() or any(
        dispatch_records[part][1] != schedule_records[part][1] for part in dispatch_records
    ):
        raise ValueError("RELION dispatch-v2 particle/order identity disagrees with its sealed schedule")
    dispatch_owner_mismatch_count = sum(
        dispatch_records[part][0] != schedule_records[part][0] for part in dispatch_records
    )
    expected_rank = np.asarray([dispatch_records[int(value)][0] for value in part_id], dtype=np.int32)
    if not np.array_equal(expected_rank, rank):
        raise ValueError("RELION summary rank ownership disagrees with dispatch-v2")
    order = np.argsort(original_identity, kind="stable")
    _validate_summary_arrays(
        original_identity[order],
        winner[order],
        runner_up[order],
        margin[order],
        class_scores[order],
        class_pose[order],
        higher_is_better=False,
        allow_tied_selected_winner=True,
    )
    metadata = dict(shared_metadata)
    metadata.update(
        {
            "engine": "relion",
            "total_bytes": total_bytes,
            "data_star_sha256": data_star_sha256,
            "input_manifest_sha256_verified": input_manifest_sha256,
            "executable_sha256_verified": executable_sha256,
            "dispatch_log_sha256": dispatch_log_sha256,
            "dispatch_schedule_sha256": dispatch_schedule_sha256,
            "dispatch_particle_order_identity_exact": True,
            "dispatch_owner_matches_recovar_oracle": dispatch_owner_mismatch_count == 0,
            "dispatch_owner_mismatch_count": dispatch_owner_mismatch_count,
            "class_counts_from_star": np.bincount(star_class, minlength=4).tolist(),
        }
    )
    return WinnerSummary(
        label=label,
        engine="relion",
        identity=original_identity[order],
        winner=winner[order],
        runner_up=runner_up[order],
        margin=margin[order],
        class_scores=class_scores[order],
        class_pose_indices=class_pose[order],
        class_second_scores=class_second_scores[order],
        class_second_pose_indices=class_second_pose[order],
        class_within_pose_margin=class_within_pose_margin[order],
        class_posterior_mass=class_posterior_mass[order],
        significant_count=significant_count[order],
        class_log_evidence=None,
        global_log_z=None,
        pose_topology=(int(nr_dir[0] * nr_psi[0]), int(nr_trans[0])),
        metadata=metadata,
        artifact_paths=shards,
    )


def _validate_summary_arrays(
    identity,
    winner,
    runner_up,
    margin,
    scores,
    poses,
    *,
    higher_is_better,
    allow_tied_selected_winner=False,
):
    n_images = identity.size
    if identity.shape != (n_images,) or np.unique(identity).size != n_images:
        raise ValueError("summary identities must be a unique vector")
    if scores.shape != (n_images, 4) or poses.shape != (n_images, 4):
        raise ValueError("summary class arrays must have shape (particles, 4)")
    if winner.shape != (n_images,) or runner_up.shape != (n_images,) or margin.shape != (n_images,):
        raise ValueError("summary winner arrays have an unexpected shape")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(margin)):
        raise ValueError("summary contains non-finite scores/margins")
    order = np.argsort(-scores if higher_is_better else scores, axis=1, kind="stable")
    if allow_tied_selected_winner:
        optimum = np.min(scores, axis=1) if not higher_is_better else np.max(scores, axis=1)
        if not np.array_equal(scores[np.arange(n_images), winner], optimum):
            raise ValueError("summary selected winner is not an optimal class score")
        masked = scores.copy()
        masked[np.arange(n_images), winner] = np.inf if not higher_is_better else -np.inf
        expected_runner = np.argmin(masked, axis=1) if not higher_is_better else np.argmax(masked, axis=1)
        if not np.array_equal(expected_runner, runner_up):
            raise ValueError("summary runner-up does not match non-winning class scores")
    elif not np.array_equal(order[:, 0], winner) or not np.array_equal(order[:, 1], runner_up):
        raise ValueError("summary winner/runner-up do not match class scores")
    expected_margin = (
        scores[np.arange(n_images), winner] - scores[np.arange(n_images), runner_up]
        if higher_is_better
        else scores[np.arange(n_images), runner_up] - scores[np.arange(n_images), winner]
    )
    if not np.array_equal(expected_margin.astype(np.float32), margin.astype(np.float32)):
        raise ValueError("summary margins do not exactly match class scores")
    if np.any(margin < 0):
        raise ValueError("summary contains negative winner margins")


def _validate_within_class_runner_up(
    best_scores,
    best_poses,
    second_scores,
    second_poses,
    within_margins,
    *,
    higher_is_better,
):
    expected_shape = best_scores.shape
    if (
        second_scores.shape != expected_shape
        or second_poses.shape != expected_shape
        or within_margins.shape != expected_shape
    ):
        raise ValueError("within-class runner-up arrays must match the class-score shape")
    if not np.all(np.isfinite(second_scores)) or not np.all(np.isfinite(within_margins)):
        raise ValueError("within-class runner-up arrays contain non-finite values")
    if np.any(best_poses == second_poses):
        raise ValueError("within-class best and second-best pose indices must differ")
    expected_margin = best_scores - second_scores if higher_is_better else second_scores - best_scores
    if not np.array_equal(expected_margin.astype(np.float32), within_margins.astype(np.float32)):
        raise ValueError("within-class pose margins do not exactly match best/second scores")
    if np.any(within_margins < 0):
        raise ValueError("within-class pose margins must be non-negative")


def read_relion_identity_classes(path: str | Path) -> tuple[dict[str, tuple[int, int, int]], np.ndarray]:
    """Return image-name -> (internal ID, original stack ID, zero-based class)."""

    loops = _read_star_loops(path)
    for columns, rows in loops:
        required = {"_rlnImageName", "_rlnClassNumber"}
        if required.issubset(columns):
            name_column = columns.index("_rlnImageName")
            class_column = columns.index("_rlnClassNumber")
            result = {}
            classes = []
            for part, row in enumerate(rows):
                name = row[name_column]
                stack_token, separator, _ = name.partition("@")
                if not separator or not stack_token.isdigit():
                    raise ValueError(f"cannot derive original identity from {name!r}")
                original = int(stack_token) - 1
                class_index = int(row[class_column]) - 1
                if name in result:
                    raise ValueError(f"duplicate RELION image name {name!r}")
                result[name] = (part, original, class_index)
                classes.append(class_index)
            return result, np.asarray(classes, dtype=np.int32)
    raise ValueError(f"no particle loop with identity/class columns in {path}")


def _read_star_loops(path: str | Path) -> list[tuple[list[str], list[list[str]]]]:
    lines = Path(path).read_text().splitlines()
    loops = []
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue
        index += 1
        columns = []
        while index < len(lines) and lines[index].strip().startswith("_rln"):
            columns.append(lines[index].split()[0])
            index += 1
        rows = []
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped or stripped.startswith(("data_", "loop_", "_rln")):
                break
            tokens = stripped.split()
            if len(tokens) != len(columns):
                raise ValueError(f"malformed STAR row in {path}: expected {len(columns)} tokens")
            rows.append(tokens)
            index += 1
        loops.append((columns, rows))
    return loops


def read_dispatch_records(path: str | Path, *, iteration: int) -> dict[int, tuple[int, int]]:
    """Return internal particle ID -> (one-based rank, sorted position)."""

    records = {}
    marker_seen = False
    with Path(path).open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                marker_seen |= "RELION_DISPATCH_LOG_SCHEMA_V2" in line
                continue
            schema, row_iteration, rank, sorted_position, part = map(int, line.split("\t"))
            if schema != 2 or row_iteration != iteration:
                continue
            if part in records:
                raise ValueError(f"duplicate dispatch identity {part}")
            records[part] = (rank, sorted_position)
    if not marker_seen or not records:
        raise ValueError("dispatch-v2 marker/rows are missing")
    positions = sorted(position for _rank, position in records.values())
    if positions != list(range(len(records))):
        raise ValueError("dispatch-v2 sorted positions are not an exact dense range")
    return records


def read_dispatch_owners(path: str | Path, *, iteration: int) -> dict[int, int]:
    return {part: rank for part, (rank, _position) in read_dispatch_records(path, iteration=iteration).items()}


def read_dispatch_schedule_records(path: str | Path, *, iteration: int) -> dict[int, tuple[int, int]]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "relion_iterations",
            "owner_by_sorted_position",
            "original_particle_id_by_sorted_position",
            "schema_version",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"dispatch schedule lacks fields: {sorted(missing)}")
        if int(_as_scalar(payload["schema_version"], "schema_version")) != 3:
            raise ValueError("dispatch schedule must use schema version 3")
        iterations = np.asarray(payload["relion_iterations"], dtype=np.int64)
        matches = np.flatnonzero(iterations == iteration)
        if matches.size != 1:
            raise ValueError(f"dispatch schedule does not contain iteration {iteration} exactly once")
        row = int(matches[0])
        owners_zero_based = np.asarray(payload["owner_by_sorted_position"][row], dtype=np.int64)
        part_ids = np.asarray(payload["original_particle_id_by_sorted_position"][row], dtype=np.int64)
    if owners_zero_based.shape != part_ids.shape or np.unique(part_ids).size != part_ids.size:
        raise ValueError("dispatch schedule has invalid topology/particle identities")
    return {
        int(part): (int(owner + 1), int(position))
        for position, (part, owner) in enumerate(zip(part_ids, owners_zero_based, strict=True))
    }


def read_dispatch_schedule_owners(path: str | Path, *, iteration: int) -> dict[int, int]:
    return {part: rank for part, (rank, _position) in read_dispatch_schedule_records(path, iteration=iteration).items()}


def analyze_summaries(summaries: list[WinnerSummary]) -> dict:
    if len(summaries) < 2:
        raise ValueError("at least two summaries are required")
    canonical_identity = summaries[0].identity
    for summary in summaries[1:]:
        if not np.array_equal(summary.identity, canonical_identity):
            raise ValueError(f"identity vector mismatch for {summary.label}")
    input_hashes = {summary.metadata.get("input_manifest_sha256") for summary in summaries}
    dispatch_hashes = {summary.metadata.get("dispatch_oracle_sha256") for summary in summaries}
    if len(input_hashes) != 1:
        raise ValueError("summaries do not share one exact input manifest")
    if len(dispatch_hashes) != 1:
        raise ValueError("summaries do not share one exact dispatch manifest")
    pose_topologies = {summary.pose_topology for summary in summaries}
    if len(pose_topologies) != 1:
        raise ValueError(f"summaries have incompatible class-local pose topologies: {sorted(pose_topologies)}")
    pose_topology = next(iter(pose_topologies))
    arms = {}
    for summary in summaries:
        arms[summary.label] = {
            "engine": summary.engine,
            "class_local_pose_topology": {
                "n_rotations": pose_topology[0],
                "n_translations": pose_topology[1],
                "flattening": "rotation_major_then_translation",
            },
            "counts": np.bincount(summary.winner, minlength=4).tolist(),
            "winner_sha256": hashlib.sha256(summary.winner.tobytes()).hexdigest(),
            "margin_quantiles": _quantiles(summary.margin),
            "within_class_pose_margin_quantiles": {
                str(class_index): _quantiles(summary.class_within_pose_margin[:, class_index])
                for class_index in range(4)
            },
            "significant_count_values": np.unique(summary.significant_count).tolist(),
            "posterior_mass_sum_by_class": np.sum(summary.class_posterior_mass, axis=0, dtype=np.float64).tolist(),
            "posterior_non_one_hot_count": int(
                np.count_nonzero(np.sum(summary.class_posterior_mass == 1.0, axis=1) != 1)
            ),
            "evidence_availability": (
                "available" if summary.class_log_evidence is not None else "unavailable_by_algorithm"
            ),
            "class_log_evidence_quantiles": (
                None
                if summary.class_log_evidence is None
                else {
                    str(class_index): _quantiles(summary.class_log_evidence[:, class_index]) for class_index in range(4)
                }
            ),
            "global_log_z_quantiles": (None if summary.global_log_z is None else _quantiles(summary.global_log_z)),
            "artifact_sha256": {str(path): sha256_file(path) for path in summary.artifact_paths},
            "metadata": summary.metadata,
        }
    pairwise = []
    for left_index, left in enumerate(summaries):
        left_losses = _centered_losses(left)
        for right in summaries[left_index + 1 :]:
            mismatches = np.flatnonzero(left.winner != right.winner)
            score_delta = left_losses - _centered_losses(right)
            confusion = np.zeros((4, 4), dtype=np.int64)
            np.add.at(confusion, (left.winner, right.winner), 1)
            score_abs_metrics = _array_metrics(np.abs(score_delta))
            margin_abs_metrics = _array_metrics(np.abs(left.margin - right.margin))
            same_engine = left.engine == right.engine
            pairwise.append(
                {
                    "left": left.label,
                    "right": right.label,
                    "mismatch_count": int(mismatches.size),
                    "mismatches": [
                        {
                            "original_index_zero_based": int(canonical_identity[index]),
                            "left_class_zero_based": int(left.winner[index]),
                            "right_class_zero_based": int(right.winner[index]),
                            "left_margin": float(left.margin[index]),
                            "right_margin": float(right.margin[index]),
                        }
                        for index in mismatches
                    ],
                    "same_engine": same_engine,
                    "raw_pose_index_interpretation": (
                        "exact same-engine index repeat control"
                        if same_engine
                        else "descriptive only; cross-engine grid-index bijection is not yet proven"
                    ),
                    "winner_confusion_matrix_left_rows_right_columns": confusion.tolist(),
                    "class_pair_signed_margin_confusion": _class_pair_sign_confusion(left, right),
                    "centered_class_loss_abs_delta": score_abs_metrics,
                    "margin_abs_delta": margin_abs_metrics,
                    "best_pose_exact_mismatches": _pose_mismatch_report(
                        left.class_pose_indices,
                        right.class_pose_indices,
                        canonical_identity,
                    ),
                    "second_pose_exact_mismatches": _pose_mismatch_report(
                        left.class_second_pose_indices,
                        right.class_second_pose_indices,
                        canonical_identity,
                    ),
                    "within_class_pose_margin_abs_delta": _array_metrics(
                        np.abs(left.class_within_pose_margin - right.class_within_pose_margin)
                    ),
                    "posterior_mass_abs_delta": _array_metrics(
                        np.abs(left.class_posterior_mass - right.class_posterior_mass)
                    ),
                    "class_log_evidence_abs_delta": (
                        None
                        if left.class_log_evidence is None or right.class_log_evidence is None
                        else _array_metrics(np.abs(left.class_log_evidence - right.class_log_evidence))
                    ),
                    "global_log_z_abs_delta": (
                        None
                        if left.global_log_z is None or right.global_log_z is None
                        else _array_metrics(np.abs(left.global_log_z - right.global_log_z))
                    ),
                    "native_float32_ulp": (
                        _ulp_metrics(left.class_scores, right.class_scores)
                        if same_engine
                        else {
                            "available": False,
                            "reason": "cross-engine raw score conventions have opposite sign and different offsets",
                        }
                    ),
                    "numerical_classification": (
                        "native_repeat_control"
                        if same_engine
                        else "unresolved_native_float32_only_requires_recomputed_float64_or_complex128_control"
                    ),
                }
            )
    repeat_floor = {}
    for engine in {summary.engine for summary in summaries}:
        matching = [
            pair
            for pair in pairwise
            if pair["same_engine"]
            and next(summary.engine for summary in summaries if summary.label == pair["left"]) == engine
        ]
        repeat_floor[engine] = {
            "centered_class_loss_max_abs": max(
                (pair["centered_class_loss_abs_delta"]["max"] for pair in matching), default=0.0
            ),
            "margin_max_abs": max((pair["margin_abs_delta"]["max"] for pair in matching), default=0.0),
        }
    engine_by_label = {summary.label: summary.engine for summary in summaries}
    for pair in pairwise:
        score_floor = max(
            repeat_floor[engine_by_label[pair["left"]]]["centered_class_loss_max_abs"],
            repeat_floor[engine_by_label[pair["right"]]]["centered_class_loss_max_abs"],
        )
        margin_floor = max(
            repeat_floor[engine_by_label[pair["left"]]]["margin_max_abs"],
            repeat_floor[engine_by_label[pair["right"]]]["margin_max_abs"],
        )
        pair["native_repeat_normalized"] = {
            "centered_class_loss_max_ratio": _safe_ratio(pair["centered_class_loss_abs_delta"]["max"], score_floor),
            "margin_max_ratio": _safe_ratio(pair["margin_abs_delta"]["max"], margin_floor),
            "score_repeat_floor": score_floor,
            "margin_repeat_floor": margin_floor,
        }
    return {
        "schema": "k4_global_winner_distribution_analysis_v1",
        "identity_count": int(canonical_identity.size),
        "identity_sha256": hashlib.sha256(canonical_identity.tobytes()).hexdigest(),
        "class_index_convention": "zero_based",
        "class_local_pose_topology": {
            "n_rotations": pose_topology[0],
            "n_translations": pose_topology[1],
            "flattening": "rotation_major_then_translation",
        },
        "metric_policy": "exact/array score and margin metrics; no correlation",
        "pose_index_comparison_policy": (
            "raw pose indices are exact same-engine repeat controls; cross-engine geometry remains unresolved "
            "until an explicit grid-index bijection is validated"
        ),
        "numerical_classification_policy": (
            "cross-engine native-float32 aggregates cannot establish numerical-noise equivalence; "
            "recomputed float64/complex128 controls are required for final classification"
        ),
        "arms": arms,
        "native_repeat_floor": repeat_floor,
        "pairwise": pairwise,
    }


def _pose_mismatch_report(
    left: np.ndarray,
    right: np.ndarray,
    identity: np.ndarray,
    *,
    sample_limit: int = 64,
) -> dict:
    if left.shape != right.shape or left.shape != (identity.size, 4):
        raise ValueError("pose mismatch reporting requires aligned (particles, 4) arrays")
    mismatch = left != right
    coordinates = np.argwhere(mismatch)
    return {
        "total_element_count": int(coordinates.shape[0]),
        "particle_any_class_count": int(np.count_nonzero(np.any(mismatch, axis=1))),
        "per_class_count": np.count_nonzero(mismatch, axis=0).astype(np.int64).tolist(),
        "sample_limit": sample_limit,
        "samples": [
            {
                "original_index_zero_based": int(identity[image_index]),
                "class_zero_based": int(class_index),
                "left_class_local_pose_index": int(left[image_index, class_index]),
                "right_class_local_pose_index": int(right[image_index, class_index]),
            }
            for image_index, class_index in coordinates[:sample_limit]
        ],
    }


def _centered_losses(summary: WinnerSummary) -> np.ndarray:
    if summary.engine == "recovar":
        return np.max(summary.class_scores, axis=1, keepdims=True) - summary.class_scores
    return summary.class_scores - np.min(summary.class_scores, axis=1, keepdims=True)


def _quantiles(array: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p001", "p01", "p10", "p50", "p90", "p99", "max"),
            np.quantile(array.astype(np.float64), (0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 1)),
            strict=True,
        )
    }


def _array_metrics(array: np.ndarray) -> dict[str, float]:
    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    return {
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "p50": float(np.quantile(flat, 0.5)),
        "p90": float(np.quantile(flat, 0.9)),
        "p99": float(np.quantile(flat, 0.99)),
        "p999": float(np.quantile(flat, 0.999)),
    }


def _class_pair_sign_confusion(left: WinnerSummary, right: WinnerSummary) -> dict:
    result = {}
    left_losses = _centered_losses(left)
    right_losses = _centered_losses(right)
    for class_a in range(4):
        for class_b in range(class_a + 1, 4):
            # Positive means class A is preferred over class B in both engines.
            left_signed = left_losses[:, class_b] - left_losses[:, class_a]
            right_signed = right_losses[:, class_b] - right_losses[:, class_a]
            left_sign = np.sign(left_signed)
            right_sign = np.sign(right_signed)
            result[f"{class_a}_vs_{class_b}"] = {
                "opposite_nonzero_sign_count": int(np.count_nonzero(left_sign * right_sign < 0)),
                "either_exact_zero_count": int(np.count_nonzero((left_sign == 0) | (right_sign == 0))),
                "left_positive_zero_negative": [
                    int(np.count_nonzero(left_sign > 0)),
                    int(np.count_nonzero(left_sign == 0)),
                    int(np.count_nonzero(left_sign < 0)),
                ],
                "right_positive_zero_negative": [
                    int(np.count_nonzero(right_sign > 0)),
                    int(np.count_nonzero(right_sign == 0)),
                    int(np.count_nonzero(right_sign < 0)),
                ],
            }
    return result


def _ordered_float32_bits(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    bits = values.view(np.uint32)
    return np.where((bits & np.uint32(0x80000000)) != 0, ~bits, bits ^ np.uint32(0x80000000)).astype(np.uint64)


def _ulp_metrics(left: np.ndarray, right: np.ndarray) -> dict:
    left32 = np.asarray(left, dtype=np.float32)
    right32 = np.asarray(right, dtype=np.float32)
    if left32.shape != right32.shape or not np.all(np.isfinite(left32)) or not np.all(np.isfinite(right32)):
        raise ValueError("ULP comparison requires same-shape finite float32 arrays")
    left_ordered = _ordered_float32_bits(left32)
    right_ordered = _ordered_float32_bits(right32)
    distance = np.where(
        left_ordered >= right_ordered,
        left_ordered - right_ordered,
        right_ordered - left_ordered,
    )
    adjacent = (left32 == right32) | (np.nextafter(left32, right32) == right32)
    return {
        "available": True,
        "max_ulp": int(np.max(distance)),
        "p50_ulp": float(np.quantile(distance, 0.5)),
        "p99_ulp": float(np.quantile(distance, 0.99)),
        "exact_fraction": float(np.mean(left32 == right32)),
        "within_one_nextafter_fraction": float(np.mean(adjacent)),
    }


def _safe_ratio(value: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return 0.0 if value == 0.0 else None
    return float(value / denominator)
