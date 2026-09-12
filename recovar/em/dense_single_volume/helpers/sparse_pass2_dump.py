"""Diagnostic dumps and group timing of the sparse bucketed pass 2.

The env-gated per-bucket dump requests and their stop/prioritization rules,
the top-2 score debug logs and the per-group timing accumulation.
``sparse_pass2_bucketed`` reports every bucket through this owner; none of
it changes production arithmetic.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from recovar.em.dense_single_volume.helpers.env_flags import parse_env_flag
from recovar.em.dense_single_volume.helpers.sparse_pass2_policy import _pass2_dump_enabled
from recovar.em.diagnostics import pass2 as pass2_diagnostics

logger = logging.getLogger(__name__)


_PASS2_DUMP_STOP_AFTER_TARGET_ENV = "RECOVAR_PASS2_DUMP_STOP_AFTER_TARGET"


_NORM_RESIDUAL_DUMP_STOP_AFTER_TARGET_ENV = (
    "RECOVAR_PASS2_DUMP_NORM_RESIDUAL_STOP_AFTER_TARGET"
)


class Pass2DumpComplete(RuntimeError):
    """Raised by explicit diagnostic runs after requested pass-2 dump files are written."""

    def __init__(self, *, dump_count: int, current_size: int | None):
        self.dump_count = int(dump_count)
        self.current_size = None if current_size is None else int(current_size)
        super().__init__(
            "requested RECOVAR pass-2 dump target set was written "
            f"(dump_count={self.dump_count}, current_size={self.current_size})"
        )


def _k_class_pass2_dump_progress(
    *,
    dump_dir: str | Path,
    target_original_indices,
    target_classes_one_based,
    current_size: int | None,
) -> tuple[int, int]:
    """Return written and expected file counts for a K-class dump target set."""

    target_indices = {int(value) for value in target_original_indices}
    target_classes = {int(value) for value in target_classes_one_based}
    if not target_indices:
        raise ValueError("K-class pass-2 dump completion requires at least one target particle")
    if not target_classes or min(target_classes) < 1:
        raise ValueError("K-class pass-2 dump completion requires positive one-based classes")
    size_label = -1 if current_size is None else int(current_size)
    root = Path(dump_dir)
    expected_paths = [
        root / f"pass2_orig{original_index:06d}_class{class_one_based:03d}_cs{size_label:03d}.npz"
        for original_index in sorted(target_indices)
        for class_one_based in sorted(target_classes)
    ]
    return sum(path.is_file() for path in expected_paths), len(expected_paths)


def _k1_pass2_dump_progress(
    *,
    dump_dir: str | Path,
    target_original_indices,
    current_size: int | None,
) -> tuple[int, int]:
    """Return written and expected file counts for a K=1 dump target set."""

    target_indices = {int(value) for value in target_original_indices}
    if not target_indices:
        raise ValueError("K=1 pass-2 dump completion requires at least one target particle")
    size_label = -1 if current_size is None else int(current_size)
    root = Path(dump_dir)
    expected_paths = [
        root / f"pass2_orig{original_index:06d}_cs{size_label:03d}.npz"
        for original_index in sorted(target_indices)
    ]
    return sum(path.is_file() for path in expected_paths), len(expected_paths)


_PASS2_TOP2_DEBUG_INDICES_ENV = "RECOVAR_PASS2_TOP2_DEBUG_INDICES"


def _pass2_top2_debug_target_indices() -> tuple[int, ...]:
    """Diagnostic only: original (combined, pre-half-split) dataset image
    indices to log the fine (pass-2) top-2 candidate score margin for,
    mirroring ``k_class._pass1_top2_debug_target_indices`` but for the
    oversampled fine-grid decision within pass-1's surviving coarse
    cell(s), where the per-particle candidate set actually differs
    (children of that particle's own coarse winner). Resolved to this
    call's local (within-half) index space via
    ``_resolve_local_target_indices`` before use -- a half-1 and a half-2
    particle can share the same local position, so matching on the raw
    env value directly would silently also hit an unrelated particle in
    the other half.
    """

    raw = os.environ.get(_PASS2_TOP2_DEBUG_INDICES_ENV, "").strip()
    if not raw:
        return ()
    return tuple(int(token) for token in raw.split(",") if token.strip())


def _resolve_local_target_indices(experiment_dataset, original_targets: tuple[int, ...]) -> tuple[int, ...]:
    """Map original (combined dataset) indices to this half's local indices.

    Only returns the subset of ``original_targets`` actually present in
    ``experiment_dataset`` (e.g. the half this call is scoring). Required
    because pass-1/pass-2 debug/override target indices are specified in
    original-dataset space but ``image_indices`` inside the per-half
    scoring functions is local (within-half) space, and two different
    halves' particles can land on the same local position.
    """

    if not original_targets:
        return ()
    resolver = getattr(experiment_dataset, "local_image_indices_from_original", None)
    if not callable(resolver):
        raise RuntimeError(
            "pass1/pass2 top-2 debug/override requires "
            "experiment_dataset.local_image_indices_from_original()"
        )
    local = np.asarray(
        resolver(np.asarray(original_targets, dtype=np.int64), allow_missing=True)
    )
    return tuple(int(v) for v in local if v >= 0)


def _log_pass2_top2_debug(scores, image_indices, targets: tuple[int, ...], *, dataset_tag=None) -> None:
    image_indices_np = np.asarray(image_indices, dtype=np.int64).reshape(-1)
    for target in targets:
        rows = np.flatnonzero(image_indices_np == target)
        if rows.size == 0:
            continue
        row = int(rows[0])
        flat = np.asarray(scores[row], dtype=np.float64).reshape(-1)
        finite = flat[np.isfinite(flat)]
        if finite.size < 1:
            logger.warning("PASS2_TOP2_DEBUG dataset=%s image_idx=%d: no finite fine candidates", dataset_tag, target)
            continue
        order = np.argsort(finite)
        best = float(finite[order[-1]])
        second = float(finite[order[-2]]) if finite.size >= 2 else float("-inf")
        n_row_trans = int(np.asarray(scores).shape[-1])
        best_flat_id = int(np.flatnonzero(flat == best)[0])
        second_candidates = np.flatnonzero(flat == second) if finite.size >= 2 else np.array([], dtype=np.int64)
        second_flat_id = int(second_candidates[0]) if second_candidates.size else -1
        logger.warning(
            "PASS2_TOP2_DEBUG dataset=%s image_idx=%d n_candidates=%d best_score=%.8f second_score=%.8f "
            "margin=%.8g best_flat_id=%d(rot=%d,trans=%d) second_flat_id=%d(rot=%d,trans=%d)",
            dataset_tag,
            target,
            finite.size,
            best,
            second,
            best - second,
            best_flat_id,
            best_flat_id // n_row_trans,
            best_flat_id % n_row_trans,
            second_flat_id,
            second_flat_id // n_row_trans if second_flat_id >= 0 else -1,
            second_flat_id % n_row_trans if second_flat_id >= 0 else -1,
        )


def _add_sparse_group_timing(group_timing: dict[str, float] | None, key: str, elapsed_s: float) -> None:
    if group_timing is None:
        return
    group_timing[key] = group_timing.get(key, 0.0) + float(elapsed_s)


def _log_sparse_kclass_group_timing(
    group_key: tuple[str, str, int],
    group_timing: dict[str, float] | None,
    *,
    wall_s: float,
) -> None:
    if group_timing is None:
        return
    build_s = group_timing.get("build", 0.0)
    fetch_s = group_timing.get("fetch", 0.0)
    prepare_s = group_timing.get("prepare", 0.0)
    score_s = group_timing.get("score", 0.0)
    mstep_noise_stats_s = group_timing.get("mstep_noise_stats", 0.0)
    mstep_weighted_sums_s = group_timing.get("mstep_weighted_sums", 0.0)
    mstep_adjoint_s = group_timing.get("mstep_adjoint", 0.0)
    noise_s = group_timing.get("noise", 0.0)
    stats_s = group_timing.get("stats", 0.0)
    total_profiled_s = build_s + fetch_s + prepare_s + score_s + mstep_noise_stats_s
    logger.info(
        "Sparse fused K-class pass-2 bucket group timing: mode=%s %s=%d "
        "build=%.2fs fetch=%.2fs prepare=%.2fs score=%.2fs "
        "mstep_noise_stats=%.2fs mstep_weighted_sums=%.2fs "
        "mstep_adjoint=%.2fs noise=%.2fs stats=%.2fs "
        "total_profiled=%.2fs wall=%.2fs",
        group_key[0],
        group_key[1],
        group_key[2],
        build_s,
        fetch_s,
        prepare_s,
        score_s,
        mstep_noise_stats_s,
        mstep_weighted_sums_s,
        mstep_adjoint_s,
        noise_s,
        stats_s,
        total_profiled_s,
        float(wall_s),
    )


def _pass2_dump_requested_for_bucket(
    *,
    experiment_dataset,
    image_indices,
    current_size,
) -> bool:
    """Return whether this bucket must stay materialized for a pass-2 dump."""

    return bool(
        pass2_diagnostics._pass2_dump_target_rows(
            experiment_dataset=experiment_dataset,
            image_indices=image_indices,
            current_size=current_size,
        ).size
    )


def _prioritize_stopped_pass2_dump_buckets(
    buckets,
    *,
    experiment_dataset,
    current_size,
):
    """Move explicitly requested dump buckets first in a stopped diagnostic.

    A stop-after-target capture consumes no M-step result, so unrelated
    particles cannot affect the requested particle's fine-score operands.
    Normal refinement and non-stopped dumps retain their original physical
    execution order.
    """

    stopped_pass2_dump = _pass2_dump_enabled() and parse_env_flag(
        _PASS2_DUMP_STOP_AFTER_TARGET_ENV, default=False
    )
    stopped_norm_dump = parse_env_flag(
        "RECOVAR_PASS2_DUMP_NORM_RESIDUAL_INPUTS", default=False
    ) and parse_env_flag(
        _NORM_RESIDUAL_DUMP_STOP_AFTER_TARGET_ENV, default=False
    )
    if not (stopped_pass2_dump or stopped_norm_dump):
        return buckets

    requested = []
    remaining = []
    for bucket in buckets:
        destination = (
            requested
            if _pass2_dump_requested_for_bucket(
                experiment_dataset=experiment_dataset,
                image_indices=bucket["image_indices"],
                current_size=current_size,
            )
            else remaining
        )
        destination.append(bucket)
    if not requested:
        return buckets
    logger.info(
        "Sparse K=1 pass-2 stopped diagnostic: moving %d requested dump "
        "bucket(s) before %d unrelated bucket(s)",
        len(requested),
        len(remaining),
    )
    return requested + remaining
