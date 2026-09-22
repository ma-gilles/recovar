"""Diagnostic dumps and group timing of the sparse bucketed pass 2.

The env-gated per-bucket dump requests and their stop/prioritization rules,
the top-2 score debug logs and the per-group timing accumulation.
``sparse_pass2_bucketed`` reports every bucket through this owner; none of
it changes production arithmetic.
"""

from __future__ import annotations

import logging
import time
import jax.numpy as jnp
from pathlib import Path

from recovar.em.diagnostics import pass2 as pass2_diagnostics
from recovar.em.helpers.env_flags import parse_env_flag
from recovar.em.sparse_pass2.sparse_pass2_policy import _pass2_dump_enabled

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


def _add_sparse_group_timing(group_timing: dict[str, float] | None, key: str, elapsed_s: float) -> None:
    if group_timing is None:
        return
    # With the sync knob the stage also absorbs the GPU work it dispatched.
    elapsed_s = float(elapsed_s) + _group_timing_device_barrier_s()
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
    prepare_substages = " ".join(
        f"{key}={group_timing.get(key, 0.0):.2f}s"
        for key in (
            "prepare_ctf_noise",
            "prepare_image_fft",
            "prepare_weighting",
            "prepare_translate",
            "prepare_window_cast",
            "score_projection_barrier",
            "mstep_active_row_sync",
            "noise_sums",
            "noise_power_shells",
            "noise_scale_correction",
            "build_kclass_arrays",
            "build_compact_pairs",
            "pipeline_throttle",
            "chunk_total",
        )
    )
    total_profiled_s = build_s + fetch_s + prepare_s + score_s + mstep_noise_stats_s
    logger.info(
        "Sparse fused K-class pass-2 bucket group timing: mode=%s %s=%d "
        "build=%.2fs fetch=%.2fs prepare=%.2fs score=%.2fs "
        "mstep_noise_stats=%.2fs mstep_weighted_sums=%.2fs "
        "mstep_adjoint=%.2fs noise=%.2fs stats=%.2fs "
        "total_profiled=%.2fs wall=%.2fs %s",
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
        prepare_substages,
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


_SPARSE_KCLASS_GROUP_TIMING_SYNC_ENV = "RECOVAR_SPARSE_KCLASS_GROUP_TIMING_SYNC"
_GROUP_TIMING_SYNC_STATE: dict[str, object] = {}

def _group_timing_device_barrier_s() -> float:
    """Wait on a default-stream diagnostic token; return the wait in seconds.

    Diagnostic only (``RECOVAR_SPARSE_KCLASS_GROUP_TIMING_SYNC=1``). JAX
    dispatches asynchronously, so a host-side stage timer otherwise charges
    the GPU work of one stage to whichever later stage first pulls a value.
    The donor uses a tiny computation as a default compute-stream fence.
    This does not establish completion of unrelated custom streams.
    """
    enabled = _GROUP_TIMING_SYNC_STATE.get("enabled")
    if enabled is None:
        enabled = parse_env_flag(_SPARSE_KCLASS_GROUP_TIMING_SYNC_ENV, default=False)
        _GROUP_TIMING_SYNC_STATE["enabled"] = enabled
    if not enabled:
        return 0.0
    token = _GROUP_TIMING_SYNC_STATE.get("token")
    if token is None:
        token = jnp.asarray(0.0, dtype=jnp.float32)
        _GROUP_TIMING_SYNC_STATE["token"] = token
    t0 = time.time()
    (token + jnp.float32(1.0)).block_until_ready()
    return time.time() - t0
