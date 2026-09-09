"""Host-side timing and bucket-progress reporting for the exact-local EM engine.

Extracted from ``local_em_engine.py`` so the engine module stays focused
on the bucket-driven EM body.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from recovar.em.dense_single_volume.helpers.env_flags import parse_env_nonnegative_int
from recovar.em.dense_single_volume.helpers.timing import TimingAccumulator

if TYPE_CHECKING:
    from recovar.em.dense_single_volume.local_layout import LocalBucketSpec

# Keep the established category for run collectors.
logger = logging.getLogger("recovar.em.dense_single_volume.local_em_engine")

EXACT_LOCAL_PROGRESS_CHUNKS_ENV = "RECOVAR_EXACT_LOCAL_PROGRESS_CHUNKS"
EXACT_LOCAL_PROGRESS_SECONDS_ENV = "RECOVAR_EXACT_LOCAL_PROGRESS_SECONDS"
DEFAULT_EXACT_LOCAL_PROGRESS_CHUNKS = 1000
DEFAULT_EXACT_LOCAL_PROGRESS_SECONDS = 300


class LocalBucketProgress:
    """Track completed local buckets and emit the established progress messages."""

    def __init__(self, bucket_specs, *, total_local_rotations: int, n_trans: int):
        progress_chunks_override = parse_env_nonnegative_int(EXACT_LOCAL_PROGRESS_CHUNKS_ENV)
        progress_seconds_override = parse_env_nonnegative_int(EXACT_LOCAL_PROGRESS_SECONDS_ENV)
        self.chunk_interval = (
            DEFAULT_EXACT_LOCAL_PROGRESS_CHUNKS if progress_chunks_override is None else int(progress_chunks_override)
        )
        self.second_interval = (
            DEFAULT_EXACT_LOCAL_PROGRESS_SECONDS
            if progress_seconds_override is None
            else int(progress_seconds_override)
        )
        self.total_chunks = len(bucket_specs)
        self.total_images = int(sum((int(bucket.image_indices.shape[0]) for bucket in bucket_specs)))
        self.completed_chunks = 0
        self.completed_images = 0
        self.started_at = time.time()
        self.last_log_at = self.started_at
        if self.total_chunks:
            logger.info(
                "Exact local bucket loop start: chunks=%d images=%d total_local_rot=%d n_trans=%d progress_chunks=%d progress_seconds=%d",
                self.total_chunks,
                self.total_images,
                total_local_rotations,
                n_trans,
                self.chunk_interval,
                self.second_interval,
            )

    def log(self, *, force: bool = False, done: bool = False) -> None:
        if not self.total_chunks:
            return
        now = time.time()
        chunk_due = (
            self.chunk_interval > 0 and self.completed_chunks > 0 and (self.completed_chunks % self.chunk_interval == 0)
        )
        time_due = (
            self.second_interval > 0
            and self.last_log_at is not None
            and (now - self.last_log_at >= float(self.second_interval))
        )
        if not (force or chunk_due or time_due):
            return
        elapsed = max(0.0, now - self.started_at)
        images_per_second = float(self.completed_images) / elapsed if elapsed > 0.0 else 0.0
        label = "done" if done else "progress"
        logger.info(
            "Exact local bucket loop %s: chunks=%d/%d images=%d/%d wall=%.1fs images/s=%.1f",
            label,
            self.completed_chunks,
            self.total_chunks,
            self.completed_images,
            self.total_images,
            elapsed,
            images_per_second,
        )
        self.last_log_at = now

    def mark_bucket_done(self, bucket: LocalBucketSpec) -> None:
        self.completed_chunks += 1
        self.completed_images += int(bucket.image_indices.shape[0])
        self.log()


_LOCAL_PREPROCESS_TIMER_KEYS = (
    "integer_shift_s",
    "translation_phase_s",
    "score_process_s",
    "recon_process_s",
    "ctf_s",
    "tile_shift_score_s",
    "tile_shift_recon_s",
    "norm_s",
    "cache_build_s",
    "cache_fetch_s",
)

_LOCAL_TRANSFER_TIMER_KEYS = (
    "reconstruction_mask_to_host_s",
    "mstep_posterior_sum_to_host_s",
    "postprocess_argmax_to_host_s",
    "postprocess_scores_to_host_s",
    "postprocess_posterior_to_host_s",
    "final_noise_to_host_s",
)

_LOCAL_TIMING_PROFILE_FIELDS = (
    ("projection_time_s", "projection_s"),
    ("big_jit_bucket_s", "big_jit_bucket_s"),
    ("fused_score_mstep_s", "fused_score_mstep_s"),
    ("local_score_s", "score_s"),
    ("local_normalize_s", "normalize_s"),
    ("local_significance_s", "significance_s"),
    ("local_mstep_s", "mstep_s"),
    ("local_pack_s", "pack_s"),
    ("local_backproject_y_s", "adjoint_y_s"),
    ("local_backproject_ctf_s", "adjoint_ctf_s"),
    ("local_noise_s", "noise_s"),
    ("local_postprocess_s", "postprocess_s"),
    ("local_host_stats_s", "host_stats_s"),
    ("local_final_accumulator_s", "final_accumulator_s"),
    ("local_stats_finalize_s", "stats_finalize_s"),
)

_LOCAL_ACCOUNTED_TIMING_SETUP_FIELDS = (
    "bucket_build_s",
    "raw_cache_build_s",
    "batch_fetch_s",
    "preprocess_s",
)

_LOCAL_ACCOUNTED_TIMING_FIELDS = _LOCAL_ACCOUNTED_TIMING_SETUP_FIELDS + tuple(
    timing_attr for _, timing_attr in _LOCAL_TIMING_PROFILE_FIELDS
)


def _new_zero_timer(keys):
    return {key: 0.0 for key in keys}


class _LocalTiming(TimingAccumulator):
    """Mutable host-side timers for one exact-local EM call."""

    def __init__(self):
        super().__init__(_LOCAL_ACCOUNTED_TIMING_FIELDS)


def _new_local_preprocess_timer():
    return _new_zero_timer(_LOCAL_PREPROCESS_TIMER_KEYS)


def _new_local_transfer_timer():
    return _new_zero_timer(_LOCAL_TRANSFER_TIMER_KEYS)


def _prefixed_timer_profile(prefix: str, timer: dict[str, float]) -> dict[str, np.float64]:
    return {f"{prefix}{key}": np.float64(value) for key, value in timer.items()}


def _local_timing_profile(timing: _LocalTiming) -> dict[str, np.float64]:
    return {
        output_key: np.float64(getattr(timing, timing_attr)) for output_key, timing_attr in _LOCAL_TIMING_PROFILE_FIELDS
    }
