"""Host-side timing accumulators for the exact-local EM engine.

Extracted from ``local_em_engine.py`` so the engine module stays focused
on the bucket-driven EM body.
"""

from __future__ import annotations

import numpy as np

from recovar.em.dense_single_volume.helpers.timing import TimingAccumulator

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
