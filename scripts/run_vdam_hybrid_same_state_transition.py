#!/usr/bin/env python3
"""Compare direct and candidate VDAM transitions from one exact live state.

The ordinary InitialModel artifacts intentionally omit the large VDAM
gradient-moment state, so independently restarted trajectories cannot prove
which implementation caused the first difference.  This diagnostic runs the
direct implementation through a requested checkpoint, retains that exact
in-memory state, and then executes a repeated one-transition panel.  The
incremental packed-final mode prewarms both packed backends before mirrored
packed-deferred/packed-final ABBA and BAAB panels.  Every arm receives
independent deep copies of the same model, particle, and sampling state.

This is a diagnostic harness, not a production continuation interface.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import datetime as dt
import gc
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

SCHEMA = "recovar.vdam_hybrid_same_state_transition.v3"
ARM_ORDER = ("direct_1", "hybrid_1", "hybrid_2", "direct_2")
FLAT_ROW_ARM_ORDER = ("direct_1", "flat_rows_1", "flat_rows_2", "direct_2")
PACKED_PROJECTION_ARM_ORDER = (
    "direct_1",
    "packed_projection_1",
    "packed_projection_2",
    "direct_2",
)
PACKED_DEFERRED_ARM_ORDER = (
    "direct_1",
    "packed_deferred_1",
    "packed_deferred_2",
    "direct_2",
)
PACKED_FINAL_NOISE_ARM_ORDER = (
    "direct_1",
    "packed_final_noise_1",
    "packed_final_noise_2",
    "direct_2",
)
PACKED_FINAL_NOISE_INCREMENTAL_ABBA_ARM_ORDER = (
    "abba_packed_deferred_1",
    "abba_packed_final_noise_1",
    "abba_packed_final_noise_2",
    "abba_packed_deferred_2",
)
PACKED_FINAL_NOISE_INCREMENTAL_BAAB_ARM_ORDER = (
    "baab_packed_final_noise_1",
    "baab_packed_deferred_1",
    "baab_packed_deferred_2",
    "baab_packed_final_noise_2",
)
PACKED_FINAL_NOISE_INCREMENTAL_ARM_ORDER = (
    *PACKED_FINAL_NOISE_INCREMENTAL_ABBA_ARM_ORDER,
    *PACKED_FINAL_NOISE_INCREMENTAL_BAAB_ARM_ORDER,
)
PACKED_FINAL_NOISE_INCREMENTAL_GATE_ARM_ORDER = (
    "direct_oracle",
    *PACKED_FINAL_NOISE_INCREMENTAL_ARM_ORDER,
)
HYBRID_IMAGE_BATCH_ABBA_ARM_ORDER = (
    "abba_batch110_1",
    "abba_batch200_1",
    "abba_batch200_2",
    "abba_batch110_2",
)
HYBRID_IMAGE_BATCH_BAAB_ARM_ORDER = (
    "baab_batch200_1",
    "baab_batch110_1",
    "baab_batch110_2",
    "baab_batch200_2",
)
HYBRID_IMAGE_BATCH_ARM_ORDER = (
    *HYBRID_IMAGE_BATCH_ABBA_ARM_ORDER,
    *HYBRID_IMAGE_BATCH_BAAB_ARM_ORDER,
)
HYBRID_IMAGE_BATCH_GATE_ARM_ORDER = (
    "direct_oracle",
    *HYBRID_IMAGE_BATCH_ARM_ORDER,
)
HYBRID_PACKED_DEFERRED_ARM_ORDER = (
    "direct_1",
    "hybrid_packed_deferred_1",
    "hybrid_packed_deferred_2",
    "direct_2",
)
STABLE_SHAPES_ARM_ORDER = (
    "stable_off_1",
    "stable_on_1",
    "stable_on_2",
    "stable_off_2",
)
STABLE_FLAT_CAPACITY_ARM_ORDER = (
    "stable_flat_off_1",
    "stable_flat_on_1",
    "stable_flat_on_2",
    "stable_flat_off_2",
)
COMPACT_POSTERIOR_ARM_ORDER = (
    "direct_1",
    "compact_posterior_1",
    "compact_posterior_2",
    "direct_2",
)
COMPACT_PACKED_DEFERRED_ARM_ORDER = (
    "direct_1",
    "compact_packed_deferred_1",
    "compact_packed_deferred_2",
    "direct_2",
)
ALL_OPTIMIZED_ARM_ORDER = (
    "direct_1",
    "all_optimized_1",
    "all_optimized_2",
    "direct_2",
)
ALL_OPTIMIZED_STABLE_SHAPES_ARM_ORDER = (
    "stable_all_off_1",
    "stable_all_on_1",
    "stable_all_on_2",
    "stable_all_off_2",
)
EXACT_COARSE_SINGLE_TRANSLATE_ARM_ORDER = (
    "single_translate_off_1",
    "single_translate_on_1",
    "single_translate_on_2",
    "single_translate_off_2",
)
EXACT_COMPACT_PREPROCESS_ARM_ORDER = (
    "compact_preprocess_off_1",
    "compact_preprocess_on_1",
    "compact_preprocess_on_2",
    "compact_preprocess_off_2",
)
FUSED_PAIR_FINE_SCORE_ARM_ORDER = (
    "pair_fine_off_1",
    "pair_fine_on_1",
    "pair_fine_on_2",
    "pair_fine_off_2",
)
FUSED_COARSE_PROJECTOR_ARM_ORDER = (
    "fused_coarse_off_1",
    "fused_coarse_on_1",
    "fused_coarse_on_2",
    "fused_coarse_off_2",
)
HYBRID_ENVIRONMENT = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID",
    "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO",
    "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE",
)
COMPACT_POSTERIOR_ENVIRONMENT = "RECOVAR_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR"
FLAT_ROW_ENVIRONMENT = "RECOVAR_INITIAL_MODEL_FLAT_LOCAL_ROWS"
STABLE_FLAT_CAPACITY_ENVIRONMENT = (
    "RECOVAR_INITIAL_MODEL_STABLE_FLAT_ROW_CAPACITY"
)
PACKED_PROJECTION_ENVIRONMENT = "RECOVAR_INITIAL_MODEL_PACKED_LOCAL_PROJECTION"
PACKED_DEFERRED_ENVIRONMENT = "RECOVAR_INITIAL_MODEL_DEFER_PACKED_VDAM"
PACKED_FINAL_NOISE_ENVIRONMENT = "RECOVAR_INITIAL_MODEL_PACKED_FINAL_NOISE"
HYBRID_IMAGE_BATCH_ENVIRONMENT = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE"
)
EXACT_COARSE_SINGLE_TRANSLATE_ENVIRONMENT = (
    "RECOVAR_K1_RELION_EXACT_COARSE_SKIP_GENERIC_OPERANDS"
)
EXACT_COARSE_ASSEMBLY_PROFILE_ENVIRONMENT = (
    "RECOVAR_K1_RELION_EXACT_COARSE_ASSEMBLY_PROFILE"
)
EXACT_COMPACT_PREPROCESS_ENVIRONMENT = (
    "RECOVAR_K1_RELION_EXACT_COMPACT_PREPROCESS"
)
FUSED_PAIR_FINE_SCORE_ENVIRONMENT = "RECOVAR_EXACT_LOCAL_FUSED_PAIR_FINE_SCORE"
FUSED_COARSE_PROJECTOR_ENVIRONMENT = "RECOVAR_K1_COARSE_FUSED_PROJECTOR"
FUSED_COARSE_FIXED_ENVIRONMENT = {
    "RECOVAR_K1_COARSE_GAUSSIAN_FFI": "1",
    "RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF": "1",
    "RECOVAR_RELION_COARSE_CANONICAL_REDUCTION": "0",
    "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION": "0",
    "RECOVAR_K1_COARSE_PREHALF_WEIGHT": "0",
    "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL": "0",
    "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS": "0",
    "RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE": "0",
}
BATCHED_POSTERIOR_ENVIRONMENT = "RECOVAR_RELION_BATCHED_POSTERIOR_PRIMITIVES"
STABLE_FOURIER_QUANTUM_ENVIRONMENT = (
    "RECOVAR_RELION_VDAM_STABLE_FOURIER_WINDOW_QUANTUM"
)
GPU_MONITOR_ENVIRONMENT = "RECOVAR_SAME_STATE_GPU_MONITOR_CSV"
PERSISTENT_CACHE_TARGET_FAMILIES = (
    "jit_run_local_bucket_big_jit",
    "jit_relion_fine_diff2_fused_translate_runtime_flat_rows_f32",
    "jit_relion_fine_diff2_fused_translate_runtime_pairs_f32",
    "jit_relion_fine_diff2_fused_translate_runtime_jobs_f32",
    "jit_relion_vdam_mstep_fused_projector_x_half",
)
HYBRID_IMAGE_BATCH_CONTROL_REQUEST = 110
HYBRID_IMAGE_BATCH_CANDIDATE_REQUEST = 500
HYBRID_IMAGE_BATCH_CONTROL_EFFECTIVE = 110
HYBRID_IMAGE_BATCH_CANDIDATE_EFFECTIVE = 200
HYBRID_IMAGE_BATCH_CONTROL_BATCH_COUNT = 2
HYBRID_IMAGE_BATCH_CANDIDATE_BATCH_COUNT = 1
HYBRID_IMAGE_BATCH_CERTIFICATE_CHUNK_ROWS = 4_608
HYBRID_IMAGE_BATCH_TRANSLATION_COUNT = 49
CANDIDATE_MODES = (
    "hybrid",
    "flat_rows",
    "packed_projection",
    "packed_deferred",
    "packed_final_noise",
    "hybrid_packed_deferred",
    "stable_shapes",
    "stable_flat_capacity",
    "compact_posterior",
    "compact_packed_deferred",
    "all_optimized",
    "all_optimized_stable_shapes",
    "exact_coarse_single_translate",
    "exact_compact_preprocess",
    "fused_pair_fine_score",
    "fused_coarse_projector",
)
META_ARRAY_KEYS = (
    "selected_particle_ids",
    "best_pose_rotation_ids",
    "pose_assignments",
    "class_assignments",
    "best_pose_translations",
    "max_posterior_per_image",
    "significant_counts",
    "cutoff_counts",
    "relion_f32_sum_weight",
    "class_posterior_sums",
    "class_direction_posterior_sums",
    "class_reconstruction_support_sums",
    "noise_sumw",
    "sigma2_offset_sumw",
    "wsum_sigma2_offset",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--acceptance-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--native-checkpoint-optimiser",
        type=Path,
        default=None,
        help="load one exact native RELION VDAM checkpoint instead of replaying RECOVAR",
    )
    parser.add_argument(
        "--native-checkpoint-data-star",
        type=Path,
        default=None,
        help="exact data STAR named by --native-checkpoint-optimiser",
    )
    parser.add_argument(
        "--native-data-dir",
        type=Path,
        default=None,
        help="particle-stack directory for a native RELION checkpoint",
    )
    parser.add_argument(
        "--target-image-name",
        default=None,
        help="unique _rlnImageName whose one-transition hard state is summarized",
    )
    parser.add_argument("--checkpoint-iteration", type=int, default=34)
    parser.add_argument("--image-batch-size", type=int, default=500)
    parser.add_argument("--candidate-mode", choices=CANDIDATE_MODES, default="hybrid")
    parser.add_argument(
        "--fused-posterior-dump-original-index",
        type=int,
        default=None,
        help="diagnostically dump one particle's production pass-2 scores in every timed arm",
    )
    parser.add_argument(
        "--coarse-prefix-dump-original-index",
        type=int,
        default=None,
        help=(
            "diagnostically dump one particle's exact production source-16 "
            "coarse operands after support is decided in every timed arm"
        ),
    )
    parser.add_argument(
        "--mirrored-incremental-panels",
        action="store_true",
        help=(
            "prewarm packed-deferred and packed-final, then measure mirrored "
            "ABBA/BAAB panels between only those two backends"
        ),
    )
    parser.add_argument(
        "--mirrored-hybrid-image-batch-panels",
        action="store_true",
        help=(
            "prewarm all ten optimized seams at image batches 110 and 200, "
            "then measure mirrored ABBA/BAAB panels from one exact state"
        ),
    )
    return parser.parse_args(argv)


def _resolve_native_checkpoint_inputs(
    *,
    optimiser: Path | None,
    data_star: Path | None,
    data_dir: Path | None,
) -> dict[str, Path] | None:
    """Resolve the all-or-none native checkpoint input contract."""

    supplied = {
        "native-checkpoint-optimiser": optimiser,
        "native-checkpoint-data-star": data_star,
        "native-data-dir": data_dir,
    }
    if not any(value is not None for value in supplied.values()):
        return None
    missing = [name for name, value in supplied.items() if value is None]
    if missing:
        raise ValueError(
            "native checkpoint mode requires all three inputs; missing "
            + ", ".join(missing)
        )
    assert optimiser is not None and data_star is not None and data_dir is not None
    resolved = {
        "optimiser": optimiser.expanduser().resolve(strict=True),
        "data_star": data_star.expanduser().resolve(strict=True),
        "data_dir": data_dir.expanduser().resolve(strict=True),
    }
    if not resolved["optimiser"].is_file():
        raise ValueError("native checkpoint optimiser must be a file")
    if not resolved["data_star"].is_file():
        raise ValueError("native checkpoint data STAR must be a file")
    if not resolved["data_dir"].is_dir():
        raise ValueError("native checkpoint data directory must be a directory")
    return resolved


def _arm_order(candidate_mode: str) -> tuple[str, str, str, str]:
    if candidate_mode == "hybrid":
        return ARM_ORDER
    if candidate_mode == "flat_rows":
        return FLAT_ROW_ARM_ORDER
    if candidate_mode == "packed_projection":
        return PACKED_PROJECTION_ARM_ORDER
    if candidate_mode == "packed_deferred":
        return PACKED_DEFERRED_ARM_ORDER
    if candidate_mode == "packed_final_noise":
        return PACKED_FINAL_NOISE_ARM_ORDER
    if candidate_mode == "hybrid_packed_deferred":
        return HYBRID_PACKED_DEFERRED_ARM_ORDER
    if candidate_mode == "stable_shapes":
        return STABLE_SHAPES_ARM_ORDER
    if candidate_mode == "stable_flat_capacity":
        return STABLE_FLAT_CAPACITY_ARM_ORDER
    if candidate_mode == "compact_posterior":
        return COMPACT_POSTERIOR_ARM_ORDER
    if candidate_mode == "compact_packed_deferred":
        return COMPACT_PACKED_DEFERRED_ARM_ORDER
    if candidate_mode == "all_optimized":
        return ALL_OPTIMIZED_ARM_ORDER
    if candidate_mode == "all_optimized_stable_shapes":
        return ALL_OPTIMIZED_STABLE_SHAPES_ARM_ORDER
    if candidate_mode == "exact_coarse_single_translate":
        return EXACT_COARSE_SINGLE_TRANSLATE_ARM_ORDER
    if candidate_mode == "exact_compact_preprocess":
        return EXACT_COMPACT_PREPROCESS_ARM_ORDER
    if candidate_mode == "fused_pair_fine_score":
        return FUSED_PAIR_FINE_SCORE_ARM_ORDER
    if candidate_mode == "fused_coarse_projector":
        return FUSED_COARSE_PROJECTOR_ARM_ORDER
    raise ValueError(f"unsupported same-state candidate mode: {candidate_mode}")


def _candidate_environment(candidate_mode: str, *, enabled: bool) -> dict[str, str]:
    values = {
        **{name: "0" for name in HYBRID_ENVIRONMENT},
        COMPACT_POSTERIOR_ENVIRONMENT: "0",
        FLAT_ROW_ENVIRONMENT: "0",
        STABLE_FLAT_CAPACITY_ENVIRONMENT: "0",
        PACKED_PROJECTION_ENVIRONMENT: "0",
        PACKED_DEFERRED_ENVIRONMENT: "0",
        PACKED_FINAL_NOISE_ENVIRONMENT: "0",
        EXACT_COARSE_SINGLE_TRANSLATE_ENVIRONMENT: "0",
        EXACT_COMPACT_PREPROCESS_ENVIRONMENT: "0",
        FUSED_PAIR_FINE_SCORE_ENVIRONMENT: "0",
        FUSED_COARSE_PROJECTOR_ENVIRONMENT: "0",
    }
    if candidate_mode == "fused_coarse_projector":
        values.update(FUSED_COARSE_FIXED_ENVIRONMENT)
        values[FUSED_COARSE_PROJECTOR_ENVIRONMENT] = "1" if enabled else "0"
        return values
    if candidate_mode == "all_optimized_stable_shapes":
        values.update({name: "1" for name in HYBRID_ENVIRONMENT})
        values[COMPACT_POSTERIOR_ENVIRONMENT] = "1"
        values[FLAT_ROW_ENVIRONMENT] = "1"
        values[STABLE_FLAT_CAPACITY_ENVIRONMENT] = "1" if enabled else "0"
        values[PACKED_PROJECTION_ENVIRONMENT] = "1"
        values[PACKED_DEFERRED_ENVIRONMENT] = "1"
        values[PACKED_FINAL_NOISE_ENVIRONMENT] = "1"
        values[EXACT_COARSE_SINGLE_TRANSLATE_ENVIRONMENT] = "1"
        values[EXACT_COMPACT_PREPROCESS_ENVIRONMENT] = "1"
        values[FUSED_PAIR_FINE_SCORE_ENVIRONMENT] = "0"
        values[HYBRID_IMAGE_BATCH_ENVIRONMENT] = "200"
        values[BATCHED_POSTERIOR_ENVIRONMENT] = "1"
        values[STABLE_FOURIER_QUANTUM_ENVIRONMENT] = "32"
        return values
    if candidate_mode in {
        "exact_coarse_single_translate",
        "exact_compact_preprocess",
        "fused_pair_fine_score",
    }:
        values.update({name: "1" for name in HYBRID_ENVIRONMENT})
        values[COMPACT_POSTERIOR_ENVIRONMENT] = "1"
        values[FLAT_ROW_ENVIRONMENT] = "1"
        values[STABLE_FLAT_CAPACITY_ENVIRONMENT] = "1"
        values[PACKED_PROJECTION_ENVIRONMENT] = "1"
        values[PACKED_DEFERRED_ENVIRONMENT] = "1"
        values[PACKED_FINAL_NOISE_ENVIRONMENT] = "1"
        values[EXACT_COARSE_SINGLE_TRANSLATE_ENVIRONMENT] = "1"
        if candidate_mode == "exact_coarse_single_translate":
            values[EXACT_COARSE_SINGLE_TRANSLATE_ENVIRONMENT] = (
                "1" if enabled else "0"
            )
        elif candidate_mode == "exact_compact_preprocess":
            values[EXACT_COMPACT_PREPROCESS_ENVIRONMENT] = (
                "1" if enabled else "0"
            )
        else:
            values[EXACT_COMPACT_PREPROCESS_ENVIRONMENT] = "1"
            values[FUSED_PAIR_FINE_SCORE_ENVIRONMENT] = "1" if enabled else "0"
        return values
    if candidate_mode == "stable_shapes":
        values.update({name: "1" for name in HYBRID_ENVIRONMENT})
        values[FLAT_ROW_ENVIRONMENT] = "1"
        values[PACKED_PROJECTION_ENVIRONMENT] = "1"
        values[PACKED_DEFERRED_ENVIRONMENT] = "1"
        return values
    if candidate_mode == "stable_flat_capacity":
        values.update({name: "1" for name in HYBRID_ENVIRONMENT})
        values[FLAT_ROW_ENVIRONMENT] = "1"
        values[STABLE_FLAT_CAPACITY_ENVIRONMENT] = "1" if enabled else "0"
        values[PACKED_PROJECTION_ENVIRONMENT] = "1"
        values[PACKED_DEFERRED_ENVIRONMENT] = "1"
        return values
    if enabled:
        if candidate_mode == "hybrid":
            values.update({name: "1" for name in HYBRID_ENVIRONMENT})
        elif candidate_mode == "flat_rows":
            values[FLAT_ROW_ENVIRONMENT] = "1"
        elif candidate_mode == "packed_projection":
            values[FLAT_ROW_ENVIRONMENT] = "1"
            values[PACKED_PROJECTION_ENVIRONMENT] = "1"
        elif candidate_mode == "packed_deferred":
            values[FLAT_ROW_ENVIRONMENT] = "1"
            values[PACKED_PROJECTION_ENVIRONMENT] = "1"
            values[PACKED_DEFERRED_ENVIRONMENT] = "1"
        elif candidate_mode == "packed_final_noise":
            values[FLAT_ROW_ENVIRONMENT] = "1"
            values[PACKED_PROJECTION_ENVIRONMENT] = "1"
            values[PACKED_DEFERRED_ENVIRONMENT] = "1"
            values[PACKED_FINAL_NOISE_ENVIRONMENT] = "1"
        elif candidate_mode == "hybrid_packed_deferred":
            values.update({name: "1" for name in HYBRID_ENVIRONMENT})
            values[FLAT_ROW_ENVIRONMENT] = "1"
            values[PACKED_PROJECTION_ENVIRONMENT] = "1"
            values[PACKED_DEFERRED_ENVIRONMENT] = "1"
        elif candidate_mode == "compact_posterior":
            values.update({name: "1" for name in HYBRID_ENVIRONMENT})
            values[COMPACT_POSTERIOR_ENVIRONMENT] = "1"
        elif candidate_mode == "compact_packed_deferred":
            values.update({name: "1" for name in HYBRID_ENVIRONMENT})
            values[COMPACT_POSTERIOR_ENVIRONMENT] = "1"
            values[FLAT_ROW_ENVIRONMENT] = "1"
            values[PACKED_PROJECTION_ENVIRONMENT] = "1"
            values[PACKED_DEFERRED_ENVIRONMENT] = "1"
        elif candidate_mode == "all_optimized":
            values.update({name: "1" for name in HYBRID_ENVIRONMENT})
            values[COMPACT_POSTERIOR_ENVIRONMENT] = "1"
            values[FLAT_ROW_ENVIRONMENT] = "1"
            values[STABLE_FLAT_CAPACITY_ENVIRONMENT] = "1"
            values[PACKED_PROJECTION_ENVIRONMENT] = "1"
            values[PACKED_DEFERRED_ENVIRONMENT] = "1"
            values[PACKED_FINAL_NOISE_ENVIRONMENT] = "1"
        else:
            raise ValueError(f"unsupported same-state candidate mode: {candidate_mode}")
    return values


def _candidate_uses_hybrid(candidate_mode: str) -> bool:
    return candidate_mode in {
        "hybrid",
        "hybrid_packed_deferred",
        "stable_shapes",
        "stable_flat_capacity",
        "compact_posterior",
        "compact_packed_deferred",
        "all_optimized",
        "all_optimized_stable_shapes",
        "exact_coarse_single_translate",
        "exact_compact_preprocess",
        "fused_pair_fine_score",
    }


def _candidate_uses_compact_posterior(candidate_mode: str) -> bool:
    return candidate_mode in {
        "compact_posterior",
        "compact_packed_deferred",
        "all_optimized",
        "all_optimized_stable_shapes",
        "exact_coarse_single_translate",
        "exact_compact_preprocess",
        "fused_pair_fine_score",
    }


def _candidate_uses_packed_deferred(candidate_mode: str) -> bool:
    return candidate_mode in {
        "compact_packed_deferred",
        "all_optimized",
        "all_optimized_stable_shapes",
        "exact_coarse_single_translate",
        "exact_compact_preprocess",
        "fused_pair_fine_score",
    }


def _arm_candidate_enabled(label: str, candidate_mode: str) -> bool:
    if candidate_mode == "all_optimized_stable_shapes":
        return label.startswith("stable_all_on_")
    if candidate_mode == "stable_shapes":
        return label.startswith("stable_on_")
    if candidate_mode == "stable_flat_capacity":
        return label.startswith("stable_flat_on_")
    if candidate_mode in {
        "exact_coarse_single_translate",
        "exact_compact_preprocess",
        "fused_pair_fine_score",
        "fused_coarse_projector",
    }:
        return "_on_" in label
    return not label.startswith("direct")


def _validate_stable_flat_capacity_profiles(
    estep_meta: dict[str, Any],
    *,
    enabled: bool,
    label: str,
) -> dict[str, Any]:
    """Prove the adapter request reached every executed local engine profile."""

    profiles: dict[str, Any] = {}
    strict_reduction_count = 0
    for key, value in sorted(estep_meta.items()):
        if not isinstance(value, dict) or "chunk_flat_score_rows" not in value:
            continue
        if not bool(value.get("flat_local_rows_enabled", False)):
            raise RuntimeError(f"{label} profile {key} did not execute flat rows")
        if "stable_flat_row_capacity_enabled" not in value:
            raise RuntimeError(
                f"{label} profile {key} omitted stable_flat_row_capacity_enabled"
            )
        if bool(value["stable_flat_row_capacity_enabled"]) != bool(enabled):
            raise RuntimeError(
                f"{label} profile {key} reported stable_flat_row_capacity_enabled="
                f"{value['stable_flat_row_capacity_enabled']!r}, expected {enabled!r}"
            )
        flat_rows = np.asarray(value["chunk_flat_score_rows"], dtype=np.int64)
        padded_rows = np.asarray(value["chunk_padded_rotations"], dtype=np.int64)
        if flat_rows.ndim != 1 or flat_rows.size == 0:
            raise RuntimeError(f"{label} profile {key} has no flat-row execution")
        if flat_rows.shape != padded_rows.shape:
            raise RuntimeError(f"{label} profile {key} row shapes do not align")
        if np.any(flat_rows > padded_rows):
            raise RuntimeError(f"{label} profile {key} exceeds the mature B*R ABI")
        if enabled and not np.array_equal(flat_rows, padded_rows):
            raise RuntimeError(f"{label} profile {key} did not use the mature B*R ABI")
        strict_reduction_count += int(np.count_nonzero(flat_rows < padded_rows))
        profiles[key] = {
            "chunk_flat_score_rows": flat_rows.tolist(),
            "chunk_padded_rotations": padded_rows.tolist(),
        }
    if not profiles:
        raise RuntimeError(f"{label} has no local engine profile to validate")
    if not enabled and strict_reduction_count == 0:
        raise RuntimeError(f"{label} stable-off arm did not retain packed row reduction")
    return {
        "enabled": bool(enabled),
        "strict_reduction_count": strict_reduction_count,
        "profiles": profiles,
    }


def _validate_stable_fourier_profiles(
    estep_meta: dict[str, Any],
    *,
    enabled: bool,
    label: str,
    image_shape: tuple[int, int],
    stable_fourier_window_quantum: int = 8,
) -> dict[str, Any]:
    """Prove logical Fourier support was carried by the requested capacity."""

    from recovar.em.dense_single_volume.helpers.fourier_window import (
        make_stable_fourier_window_shape_plan,
    )

    image_shape = tuple(int(value) for value in image_shape)
    if len(image_shape) != 2 or image_shape[0] != image_shape[1]:
        raise RuntimeError(f"{label} has invalid image shape {image_shape}")
    stable_fourier_window_quantum = int(stable_fourier_window_quantum)
    if (
        stable_fourier_window_quantum < 2
        or stable_fourier_window_quantum % 2 != 0
    ):
        raise RuntimeError(
            f"{label} has invalid stable Fourier quantum "
            f"{stable_fourier_window_quantum}, expected an even integer >= 2"
        )
    for key in (
        "requested_stable_fourier_window_shapes",
        "effective_stable_fourier_window_shapes",
    ):
        if key not in estep_meta:
            raise RuntimeError(f"{label} did not report {key}")
        if bool(estep_meta[key]) != bool(enabled):
            raise RuntimeError(
                f"{label} reported {key}={estep_meta[key]!r}, expected {enabled!r}"
            )

    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    profiles: dict[str, Any] = {}
    for key, value in sorted(estep_meta.items()):
        if not isinstance(value, dict) or "chunk_padded_rotations" not in value:
            continue
        required = (
            "stable_fourier_window_shapes",
            "stable_fourier_window_quantum",
            "logical_current_size",
            "physical_current_size",
            "logical_reconstruction_pixels",
            "physical_reconstruction_pixels",
            "n_windowed",
            "n_projection_windowed",
            "big_jit_projection_pixels",
        )
        missing = [field for field in required if field not in value]
        if missing:
            raise RuntimeError(
                f"{label} profile {key} omitted stable Fourier fields {missing}"
            )
        logical_size = int(value["logical_current_size"])
        plan = make_stable_fourier_window_shape_plan(
            image_shape,
            logical_size,
            n_half,
            enabled=bool(enabled),
            quantum=stable_fourier_window_quantum,
            recon_exact_radius=False,
        )
        expected = {
            "stable_fourier_window_shapes": bool(
                enabled and plan.logical_spec.use_window
            ),
            "stable_fourier_window_quantum": stable_fourier_window_quantum,
            "logical_current_size": int(plan.logical_current_size),
            "physical_current_size": int(plan.physical_current_size),
            "logical_reconstruction_pixels": int(
                plan.logical_reconstruction_pixels
            ),
            "physical_reconstruction_pixels": int(
                plan.physical_reconstruction_pixels
            ),
            "n_windowed": int(plan.physical_score_pixels),
            "n_projection_windowed": int(plan.physical_projection_pixels),
            "big_jit_projection_pixels": int(plan.physical_projection_pixels),
        }
        observed = {field: _json_ready(value[field]) for field in expected}
        for field, expected_value in expected.items():
            if observed[field] != expected_value:
                raise RuntimeError(
                    f"{label} profile {key} reported {field}="
                    f"{observed[field]!r}, expected {expected_value!r}"
                )
        profiles[key] = observed
    if not profiles:
        raise RuntimeError(f"{label} has no local engine profile to validate")
    return {
        "enabled": bool(enabled),
        "image_shape": list(image_shape),
        "stable_fourier_window_quantum": stable_fourier_window_quantum,
        "profiles": profiles,
    }


def _validate_stable_coarse_square_profiles(
    estep_meta: dict[str, Any],
    *,
    enabled: bool,
    label: str,
    image_size: int,
    stable_fourier_window_quantum: int,
) -> dict[str, Any]:
    """Prove coarse significance used the same physical/logical size policy."""

    from recovar.em.dense_single_volume.helpers.fourier_window import (
        stable_fourier_window_current_size,
    )

    profiles: dict[str, Any] = {}
    for key, hybrid in sorted(_coarse_hybrid_profiles(estep_meta).items()):
        layout = hybrid.get("coarse_square_layout")
        if not isinstance(layout, dict):
            raise RuntimeError(
                f"{label} profile {key} omitted coarse_square_layout"
            )
        logical_size = int(layout.get("logical_current_size", -1))
        physical_size = stable_fourier_window_current_size(
            logical_size,
            int(image_size),
            quantum=int(stable_fourier_window_quantum),
        ) if enabled else logical_size
        expected = {
            "stable_fourier_window_shapes_requested": bool(enabled),
            "stable_fourier_window_shapes_effective": bool(
                enabled and physical_size != logical_size
            ),
            "logical_current_size": logical_size,
            "physical_current_size": physical_size,
            "logical_square_pixels": logical_size * (logical_size // 2 + 1),
            "physical_square_pixels": physical_size * (physical_size // 2 + 1),
            "executed_square_pixels": logical_size * (logical_size // 2 + 1),
            "logical_issue_stream_is_prefix": True,
            "physical_tail_zero_weighted": True,
            "physical_tail_skipped_by_runtime_count": bool(
                enabled and physical_size != logical_size
            ),
        }
        observed = {field: _json_ready(layout.get(field)) for field in expected}
        if observed != expected:
            raise RuntimeError(
                f"{label} profile {key} coarse stable-square mismatch: "
                f"observed={observed!r}, expected={expected!r}"
            )
        profiles[key] = observed
    if not profiles:
        raise RuntimeError(f"{label} has no coarse hybrid profile to validate")
    return {
        "enabled": bool(enabled),
        "stable_fourier_window_quantum": int(stable_fourier_window_quantum),
        "profile_exact": True,
        "profiles": profiles,
    }


ALL_OPTIMIZED_SEAMS = (
    "certified_coarse_hybrid",
    "coarse_gemm_macro",
    "coarse_projection_cache",
    "compact_posterior",
    "flat_local_rows",
    "packed_local_projection",
    "packed_vdam_deferral",
    "stable_fourier_window_shapes",
    "stable_coarse_significance_shapes",
    "stable_flat_row_capacity",
    "packed_final_noise",
)


def _validate_all_optimized_profiles(
    estep_meta: dict[str, Any],
    *,
    enabled: bool,
    label: str,
    image_shape: tuple[int, int],
    stable_fourier_window_quantum: int = 8,
) -> dict[str, Any]:
    """Fail closed unless every seam in the composed arm is effective."""

    stable_fourier = _validate_stable_fourier_profiles(
        estep_meta,
        enabled=enabled,
        label=label,
        image_shape=image_shape,
        stable_fourier_window_quantum=stable_fourier_window_quantum,
    )
    stable_coarse = (
        _validate_stable_coarse_square_profiles(
            estep_meta,
            enabled=True,
            label=label,
            image_size=int(image_shape[0]),
            stable_fourier_window_quantum=stable_fourier_window_quantum,
        )
        if enabled
        else None
    )
    for key in (
        "requested_stable_flat_row_capacity",
        "effective_stable_flat_row_capacity",
    ):
        if key not in estep_meta:
            raise RuntimeError(f"{label} did not report {key}")
        if bool(estep_meta[key]) != bool(enabled):
            raise RuntimeError(
                f"{label} reported {key}={estep_meta[key]!r}, expected {enabled!r}"
            )

    expected_flags = {
        "flat_local_rows_enabled": bool(enabled),
        "stable_flat_row_capacity_enabled": bool(enabled),
        "packed_local_projection_enabled": bool(enabled),
        "defer_packed_vdam_enabled": bool(enabled),
        "packed_vdam_reuses_flat_score_projection": bool(enabled),
    }
    local_profiles: dict[str, Any] = {}
    for key, value in sorted(estep_meta.items()):
        if not isinstance(value, dict) or "chunk_padded_rotations" not in value:
            continue
        missing = [field for field in expected_flags if field not in value]
        if missing:
            raise RuntimeError(
                f"{label} profile {key} omitted composed seam fields {missing}"
            )
        observed = {
            field: bool(value[field]) for field in expected_flags
        }
        for field, expected in expected_flags.items():
            if observed[field] is not expected:
                raise RuntimeError(
                    f"{label} profile {key} reported {field}="
                    f"{observed[field]!r}, expected {expected!r}"
                )
        local_profiles[key] = observed
    if not local_profiles:
        raise RuntimeError(f"{label} has no local engine profile to validate")

    stable_flat = None
    if enabled:
        stable_flat = _validate_stable_flat_capacity_profiles(
            estep_meta,
            enabled=True,
            label=label,
        )
    packed_final_noise = _validate_packed_final_noise_profiles(
        estep_meta,
        backend_mode="packed_final_noise" if enabled else "direct",
    )
    return {
        "enabled": bool(enabled),
        "enabled_seams": list(ALL_OPTIMIZED_SEAMS if enabled else ()),
        "disabled_seams": list(() if enabled else ALL_OPTIMIZED_SEAMS),
        "profile_exact": True,
        "stable_fourier": stable_fourier,
        "stable_coarse_significance": stable_coarse,
        "stable_flat_capacity": stable_flat,
        "packed_final_noise": packed_final_noise,
        "local_profiles": local_profiles,
    }


def _validate_all_optimized_stable_pair_profiles(
    estep_meta: dict[str, Any],
    *,
    enabled: bool,
    label: str,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    """Validate the q32/stable-flat pair while every other seam stays on."""

    stable_fourier = _validate_stable_fourier_profiles(
        estep_meta,
        enabled=enabled,
        label=label,
        image_shape=image_shape,
        # The inactive adapter intentionally reports its canonical default
        # quantum; q32 becomes an effective ABI only when stable shapes are on.
        stable_fourier_window_quantum=32 if enabled else 8,
    )
    stable_coarse = _validate_stable_coarse_square_profiles(
        estep_meta,
        enabled=enabled,
        label=label,
        image_size=int(image_shape[0]),
        stable_fourier_window_quantum=32 if enabled else 8,
    )
    for key in (
        "requested_stable_flat_row_capacity",
        "effective_stable_flat_row_capacity",
    ):
        if key not in estep_meta or bool(estep_meta[key]) != bool(enabled):
            raise RuntimeError(
                f"{label} reported {key}={estep_meta.get(key)!r}, "
                f"expected {enabled!r}"
            )

    expected_flags = {
        "flat_local_rows_enabled": True,
        "stable_flat_row_capacity_enabled": bool(enabled),
        "packed_local_projection_enabled": True,
        "defer_packed_vdam_enabled": True,
        "packed_vdam_reuses_flat_score_projection": True,
    }
    local_profiles: dict[str, Any] = {}
    for key, value in sorted(estep_meta.items()):
        if not isinstance(value, dict) or "chunk_padded_rotations" not in value:
            continue
        missing = [field for field in expected_flags if field not in value]
        if missing:
            raise RuntimeError(
                f"{label} profile {key} omitted stable-pair fields {missing}"
            )
        observed = {field: bool(value[field]) for field in expected_flags}
        if observed != expected_flags:
            raise RuntimeError(
                f"{label} stable-pair local profile differs: "
                f"observed={observed}, expected={expected_flags}"
            )
        local_profiles[key] = observed
    if not local_profiles:
        raise RuntimeError(f"{label} has no local engine profile to validate")

    stable_flat = _validate_stable_flat_capacity_profiles(
        estep_meta,
        enabled=enabled,
        label=label,
    )
    packed_final_noise = _validate_packed_final_noise_profiles(
        estep_meta,
        backend_mode="packed_final_noise",
    )
    return {
        "stable_representation_enabled": bool(enabled),
        "all_other_optimized_seams_enabled": True,
        "batched_posterior_primitives_enabled": True,
        "hybrid_image_batch_size": 200,
        "exact_coarse_skip_generic_operands": True,
        "exact_compact_preprocess": True,
        "fused_pair_fine_score": False,
        "profile_exact": True,
        "stable_fourier": stable_fourier,
        "stable_coarse_significance": stable_coarse,
        "stable_flat_capacity": stable_flat,
        "packed_final_noise": packed_final_noise,
        "local_profiles": local_profiles,
    }


def _coarse_hybrid_profiles(meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Collect every per-halfset certified-hybrid execution profile."""

    result: dict[str, dict[str, Any]] = {}
    for key, value in meta.items():
        if not isinstance(value, dict) or "coarse_gaussian_gemm_hybrid" not in value:
            continue
        profile = value["coarse_gaussian_gemm_hybrid"]
        if not isinstance(profile, dict):
            raise RuntimeError(f"{key} hybrid execution profile is not a mapping")
        result[key] = profile
    return result


def _validate_exact_coarse_single_translate_profiles(
    estep_meta: dict[str, Any],
    *,
    enabled: bool,
    label: str,
) -> dict[str, Any]:
    """Prove the exact path issued one rather than two translations per batch."""

    expected_skipped_outputs = (
        [
            "coarse_gaussian_shifted_corrected",
            "coarse_gaussian_pixel_weight",
            "coarse_gaussian_unshifted_corrected",
        ]
        if enabled
        else []
    )
    profiles: dict[str, Any] = {}
    total_batch_count = 0
    total_translate_score_call_count = 0
    for key, value in sorted(estep_meta.items()):
        if not isinstance(value, dict):
            continue
        profile = value.get("exact_coarse_operand_assembly")
        if profile is None:
            continue
        if not isinstance(profile, dict):
            raise RuntimeError(
                f"{label} profile {key} exact-coarse assembly is not a mapping",
            )
        hybrid = value.get("coarse_gaussian_gemm_hybrid")
        if not isinstance(hybrid, dict) or not isinstance(
            hybrid.get("batch_count"),
            int,
        ):
            raise RuntimeError(
                f"{label} profile {key} omitted the matching hybrid batch count",
            )
        batch_count = int(hybrid["batch_count"])
        expected = {
            "skip_generic_default_enabled": False,
            "skip_generic_requested": bool(enabled),
            "skip_generic_effective": bool(enabled),
            "exact_coarse_operands_effective": True,
            "generic_assembly_count": 0 if enabled else batch_count,
            "exact_assembly_count": batch_count,
            "translate_score_call_site_count": (
                batch_count if enabled else 2 * batch_count
            ),
            "translate_score_call_count": (
                batch_count if enabled else 2 * batch_count
            ),
            "downstream_operand_source": "exact_source_star",
            "diagnostic_operand_source": "exact_source_star",
            "raw_score_capture_changed": False,
            "skipped_generic_outputs": expected_skipped_outputs,
        }
        observed = {field: _json_ready(profile.get(field)) for field in expected}
        if observed != expected:
            raise RuntimeError(
                f"{label} profile {key} exact-coarse assembly mismatch: "
                f"observed={observed!r}, expected={expected!r}",
            )
        profiles[key] = {
            "batch_count": batch_count,
            **observed,
        }
        total_batch_count += batch_count
        total_translate_score_call_count += int(
            observed["translate_score_call_count"],
        )
    if not profiles:
        raise RuntimeError(
            f"{label} did not publish an exact-coarse operand profile",
        )
    return {
        "enabled": bool(enabled),
        "profile_exact": True,
        "profile_count": len(profiles),
        "total_batch_count": total_batch_count,
        "total_translate_score_call_count": total_translate_score_call_count,
        "expected_calls_per_batch": 1 if enabled else 2,
        "wall_timing_policy": (
            "outer transition wall; profile counters add no device synchronization"
        ),
        "profile_counter_device_synchronization": False,
        "profiles": profiles,
    }


def _validate_exact_compact_preprocess_profiles(
    estep_meta: dict[str, Any],
    *,
    enabled: bool,
    label: str,
) -> dict[str, Any]:
    """Prove exact compact scoring removed only the overwritten preprocess."""

    profiles: dict[str, Any] = {}
    total_batch_count = 0
    total_generic_score_preprocess_count = 0
    total_exact_source_preprocess_count = 0
    skipped_outputs = [
        "coarse_gaussian_shifted_corrected",
        "coarse_gaussian_pixel_weight",
        "coarse_gaussian_unshifted_corrected",
    ]
    for key, value in sorted(estep_meta.items()):
        if not isinstance(value, dict):
            continue
        profile = value.get("exact_coarse_operand_assembly")
        if profile is None:
            continue
        if not isinstance(profile, dict):
            raise RuntimeError(
                f"{label} profile {key} exact-coarse assembly is not a mapping",
            )
        hybrid = value.get("coarse_gaussian_gemm_hybrid")
        if not isinstance(hybrid, dict) or not isinstance(
            hybrid.get("batch_count"),
            int,
        ):
            raise RuntimeError(
                f"{label} profile {key} omitted the matching hybrid batch count",
            )
        batch_count = int(hybrid["batch_count"])
        generic_count = 0 if enabled else batch_count
        expected = {
            "skip_generic_default_enabled": False,
            "skip_generic_requested": True,
            "skip_generic_effective": True,
            "exact_coarse_operands_effective": True,
            "exact_compact_preprocess_default_enabled": False,
            "exact_compact_preprocess_requested": bool(enabled),
            "exact_compact_preprocess_effective": bool(enabled),
            "generic_score_preprocess_count": generic_count,
            "exact_source_preprocess_count": batch_count,
            "generic_ctf_evaluation_count": generic_count,
            "generic_full_translation_count": generic_count,
            "generic_assembly_count": 0,
            "exact_assembly_count": batch_count,
            "translate_score_call_site_count": batch_count,
            "translate_score_call_count": batch_count,
            "downstream_operand_source": "exact_source_star",
            "diagnostic_operand_source": "exact_source_star",
            "raw_score_capture_changed": False,
            "generic_fallback_policy": (
                "exact_full_direct_scores_available" if enabled else "available"
            ),
            "skipped_generic_outputs": skipped_outputs,
        }
        observed = {field: _json_ready(profile.get(field)) for field in expected}
        if observed != expected:
            raise RuntimeError(
                f"{label} profile {key} exact compact preprocessing mismatch: "
                f"observed={observed!r}, expected={expected!r}",
            )
        profiles[key] = {"batch_count": batch_count, **observed}
        total_batch_count += batch_count
        total_generic_score_preprocess_count += generic_count
        total_exact_source_preprocess_count += batch_count
    if not profiles:
        raise RuntimeError(
            f"{label} did not publish an exact compact preprocessing profile",
        )
    return {
        "enabled": bool(enabled),
        "profile_exact": True,
        "profile_count": len(profiles),
        "total_batch_count": total_batch_count,
        "total_generic_score_preprocess_count": (
            total_generic_score_preprocess_count
        ),
        "total_exact_source_preprocess_count": total_exact_source_preprocess_count,
        "expected_process_half_image_calls_per_batch": 1 if enabled else 2,
        "wall_timing_policy": (
            "outer transition wall; profile counters add no device synchronization"
        ),
        "profile_counter_device_synchronization": False,
        "profiles": profiles,
    }


def _validate_fused_pair_fine_profiles(
    estep_meta: dict[str, Any],
    *,
    enabled: bool,
    label: str,
) -> dict[str, Any]:
    """Prove the shared compact-pair ABI reached every exact-local profile."""

    for key in (
        "requested_fused_pair_fine_score",
        "effective_fused_pair_fine_score",
    ):
        if key not in estep_meta:
            raise RuntimeError(f"{label} did not report {key}")
        if bool(estep_meta[key]) != bool(enabled):
            raise RuntimeError(
                f"{label} reported {key}={estep_meta[key]!r}, expected {enabled!r}",
            )

    expected_flags = {
        "fused_pair_fine_score_enabled": bool(enabled),
        "fused_pair_fine_score_default_enabled": False,
        "fused_pair_fine_uses_shared_compact_order": bool(enabled),
        "fused_pair_fine_avoids_pair_pixel_gathers": bool(enabled),
        "fused_pair_fine_restores_dense_posterior_order": bool(enabled),
    }
    profiles: dict[str, Any] = {}
    capacity_families: set[int] = set()
    total_candidates = 0
    total_capacity = 0
    total_dense_capacity = 0
    for key, value in sorted(estep_meta.items()):
        if not isinstance(value, dict) or "chunk_padded_rotations" not in value:
            continue
        missing = [field for field in expected_flags if field not in value]
        if missing:
            raise RuntimeError(
                f"{label} profile {key} omitted fused-pair flags {missing}",
            )
        observed_flags = {field: bool(value[field]) for field in expected_flags}
        for field, expected in expected_flags.items():
            if observed_flags[field] is not expected:
                raise RuntimeError(
                    f"{label} profile {key} reported {field}="
                    f"{observed_flags[field]!r}, expected {expected!r}",
                )

        array_fields = (
            "chunk_fused_pair_capacities",
            "chunk_fused_pair_counts",
            "chunk_fused_pair_dense_capacities",
        )
        missing = [field for field in array_fields if field not in value]
        if missing:
            raise RuntimeError(
                f"{label} profile {key} omitted fused-pair arrays {missing}",
            )
        capacities, counts, dense_capacities = (
            np.asarray(value[field], dtype=np.int64) for field in array_fields
        )
        if not (
            capacities.ndim == counts.ndim == dense_capacities.ndim == 1
            and capacities.shape == counts.shape == dense_capacities.shape
        ):
            raise RuntimeError(f"{label} profile {key} fused-pair arrays do not align")

        sum_fields = (
            "sum_fused_pair_candidates",
            "sum_fused_pair_capacity",
            "sum_fused_pair_dense_capacity",
            "fused_pair_valid_fraction_of_dense",
            "fused_pair_padded_fraction_of_dense",
        )
        missing = [field for field in sum_fields if field not in value]
        if missing:
            raise RuntimeError(
                f"{label} profile {key} omitted fused-pair totals {missing}",
            )
        candidates = int(value["sum_fused_pair_candidates"])
        capacity = int(value["sum_fused_pair_capacity"])
        dense_capacity = int(value["sum_fused_pair_dense_capacity"])
        valid_fraction = float(value["fused_pair_valid_fraction_of_dense"])
        padded_fraction = float(value["fused_pair_padded_fraction_of_dense"])
        if enabled:
            if capacities.size == 0 or np.any(capacities <= 0):
                raise RuntimeError(f"{label} profile {key} has no pair capacity")
            if (
                np.any(counts < 0)
                or np.any(counts > capacities)
                or np.any(capacities > dense_capacities)
                or np.any(dense_capacities <= 0)
            ):
                raise RuntimeError(f"{label} profile {key} has invalid pair counts")
            observed_sums = (
                int(np.sum(counts, dtype=np.int64)),
                int(np.sum(capacities, dtype=np.int64)),
                int(np.sum(dense_capacities, dtype=np.int64)),
            )
            published_sums = (candidates, capacity, dense_capacity)
            if observed_sums != published_sums:
                raise RuntimeError(
                    f"{label} profile {key} fused-pair chunk sums "
                    f"{observed_sums} do not match published totals "
                    f"{published_sums}",
                )
            if not (0 < candidates <= capacity <= dense_capacity):
                raise RuntimeError(
                    f"{label} profile {key} has invalid fused-pair totals: "
                    f"{candidates}, {capacity}, {dense_capacity}",
                )
            expected_valid = candidates / dense_capacity
            expected_padded = capacity / dense_capacity
            if not np.isclose(valid_fraction, expected_valid, rtol=1e-15, atol=0.0):
                raise RuntimeError(
                    f"{label} profile {key} valid-pair fraction is inconsistent",
                )
            if not np.isclose(padded_fraction, expected_padded, rtol=1e-15, atol=0.0):
                raise RuntimeError(
                    f"{label} profile {key} padded-pair fraction is inconsistent",
                )
            capacity_families.update(int(item) for item in capacities.tolist())
        elif capacities.size or counts.size or dense_capacities.size:
            raise RuntimeError(
                f"{label} profile {key} disabled fused-pair arm published chunks",
            )
        elif any((candidates, capacity, dense_capacity, valid_fraction, padded_fraction)):
            raise RuntimeError(
                f"{label} profile {key} disabled fused-pair arm published totals",
            )

        profiles[key] = {
            **observed_flags,
            "chunk_count": int(capacities.size),
            "capacity_families": sorted(set(int(item) for item in capacities.tolist())),
            "sum_candidates": candidates,
            "sum_capacity": capacity,
            "sum_dense_capacity": dense_capacity,
            "valid_fraction_of_dense": valid_fraction,
            "padded_fraction_of_dense": padded_fraction,
        }
        total_candidates += candidates
        total_capacity += capacity
        total_dense_capacity += dense_capacity
    if not profiles:
        raise RuntimeError(f"{label} has no local engine profile to validate")
    return {
        "enabled": bool(enabled),
        "profile_exact": True,
        "profile_count": len(profiles),
        "shared_compact_pair_order": bool(enabled),
        "pair_pixel_gathers_materialized": False,
        "dense_posterior_order_restored": bool(enabled),
        "compile_shape_capacity_families": sorted(capacity_families),
        "compile_shape_capacity_family_count": len(capacity_families),
        "sum_candidates": total_candidates,
        "sum_capacity": total_capacity,
        "sum_dense_capacity": total_dense_capacity,
        "profiles": profiles,
    }


def _validate_hybrid_image_batch_profiles(
    estep_meta: dict[str, Any],
    *,
    requested_batch_size: int,
    label: str,
) -> dict[str, Any]:
    """Prove that only the compact-hybrid image row batch changed."""

    expected_by_request = {
        HYBRID_IMAGE_BATCH_CONTROL_REQUEST: (
            HYBRID_IMAGE_BATCH_CONTROL_EFFECTIVE,
            HYBRID_IMAGE_BATCH_CONTROL_BATCH_COUNT,
        ),
        HYBRID_IMAGE_BATCH_CANDIDATE_REQUEST: (
            HYBRID_IMAGE_BATCH_CANDIDATE_EFFECTIVE,
            HYBRID_IMAGE_BATCH_CANDIDATE_BATCH_COUNT,
        ),
    }
    try:
        effective_batch_size, expected_batch_count = expected_by_request[
            int(requested_batch_size)
        ]
    except KeyError as error:
        raise RuntimeError(
            f"unsupported hybrid image-batch request {requested_batch_size!r}",
        ) from error
    expected = {
        "input_image_batch_size": HYBRID_IMAGE_BATCH_CONTROL_EFFECTIVE,
        "requested_hybrid_image_batch_size": int(requested_batch_size),
        "effective_image_batch_size": effective_batch_size,
        "streamed_certificate_candidate_count_at_effective_batch": (
            effective_batch_size
            * HYBRID_IMAGE_BATCH_CERTIFICATE_CHUNK_ROWS
            * HYBRID_IMAGE_BATCH_TRANSLATION_COUNT
        ),
        "batch_count": expected_batch_count,
        "selected_rescore_batch_count": expected_batch_count,
        "fallback_batch_count": 0,
        "fallback_image_count": 0,
        "certificate_chunk_count_per_batch": 8,
        "certificate_chunk_rows": HYBRID_IMAGE_BATCH_CERTIFICATE_CHUNK_ROWS,
        "selected_rescore_image_count": HYBRID_IMAGE_BATCH_CANDIDATE_EFFECTIVE,
        "actual_image_batch_sizes": (
            [110, 90]
            if effective_batch_size == HYBRID_IMAGE_BATCH_CONTROL_EFFECTIVE
            else [HYBRID_IMAGE_BATCH_CANDIDATE_EFFECTIVE]
        ),
        "physical_image_batch_sizes": (
            [110, 110]
            if effective_batch_size == HYBRID_IMAGE_BATCH_CONTROL_EFFECTIVE
            else [HYBRID_IMAGE_BATCH_CANDIDATE_EFFECTIVE]
        ),
    }
    profiles = _coarse_hybrid_profiles(estep_meta)
    if not profiles:
        raise RuntimeError(f"{label} did not publish a hybrid image-batch profile")
    observed_profiles: dict[str, Any] = {}
    for profile_name, profile in profiles.items():
        observed: dict[str, Any] = {}
        for field, expected_value in expected.items():
            if field not in profile:
                raise RuntimeError(
                    f"{label} profile {profile_name} omitted {field}",
                )
            observed[field] = _json_ready(profile[field])
            if observed[field] != expected_value:
                raise RuntimeError(
                    f"{label} profile {profile_name} reported {field}="
                    f"{observed[field]!r}, expected {expected_value!r}",
                )
        observed_profiles[profile_name] = observed
    return {
        "requested_batch_size": int(requested_batch_size),
        "effective_batch_size": int(effective_batch_size),
        "profile_exact": True,
        "profiles": observed_profiles,
        "pass1_image_shape_variants": {
            "actual": sorted(set(expected["actual_image_batch_sizes"])),
            "physical": sorted(set(expected["physical_image_batch_sizes"])),
            "actual_variant_count": len(set(expected["actual_image_batch_sizes"])),
            "physical_variant_count": len(
                set(expected["physical_image_batch_sizes"]),
            ),
        },
    }


def _validate_packed_final_noise_profiles(
    estep_meta: dict[str, Any],
    *,
    backend_mode: str,
) -> dict[str, Any]:
    """Prove each local engine profile executed the named noise backend."""

    expected_by_backend = {
        "direct": {
            "flat_local_rows_enabled": False,
            "packed_local_projection_enabled": False,
            "defer_packed_vdam_enabled": False,
            "packed_vdam_reuses_flat_score_projection": False,
            "packed_final_noise_enabled": False,
            "packed_vdam_avoids_dense_noise_rows": False,
            "packed_final_noise_preserves_dense_scalar_order": False,
        },
        "packed_deferred": {
            "flat_local_rows_enabled": True,
            "packed_local_projection_enabled": True,
            "defer_packed_vdam_enabled": True,
            "packed_vdam_reuses_flat_score_projection": True,
            "packed_final_noise_enabled": False,
            "packed_vdam_avoids_dense_noise_rows": False,
            "packed_final_noise_preserves_dense_scalar_order": False,
        },
        "packed_final_noise": {
            "flat_local_rows_enabled": True,
            "packed_local_projection_enabled": True,
            "defer_packed_vdam_enabled": True,
            "packed_vdam_reuses_flat_score_projection": True,
            "packed_final_noise_enabled": True,
            "packed_vdam_avoids_dense_noise_rows": True,
            "packed_final_noise_preserves_dense_scalar_order": True,
        },
    }
    try:
        expected_flags = expected_by_backend[backend_mode]
    except KeyError as error:
        raise RuntimeError(
            f"unsupported packed-final gate backend {backend_mode!r}",
        ) from error

    profiles: dict[str, Any] = {}
    for key, value in sorted(estep_meta.items()):
        if not isinstance(value, dict) or "chunk_padded_rotations" not in value:
            continue
        missing = [
            field
            for field in (*expected_flags, "sum_packed_final_noise_rows")
            if field not in value
        ]
        if missing:
            raise RuntimeError(
                f"{backend_mode} profile {key} omitted packed-noise fields {missing}",
            )
        observed_flags = {
            field: bool(value[field]) for field in expected_flags
        }
        for field, expected in expected_flags.items():
            if observed_flags[field] is not expected:
                raise RuntimeError(
                    f"{backend_mode} profile {key} reported {field}="
                    f"{observed_flags[field]!r}, expected {expected!r}",
                )
        packed_rows = int(value["sum_packed_final_noise_rows"])
        if backend_mode == "packed_final_noise":
            if packed_rows <= 0:
                raise RuntimeError(
                    f"{backend_mode} profile {key} reported no packed final-noise rows",
                )
        elif packed_rows != 0:
            raise RuntimeError(
                f"{backend_mode} profile {key} unexpectedly reported "
                f"{packed_rows} packed final-noise rows",
            )
        profiles[key] = {
            **observed_flags,
            "sum_packed_final_noise_rows": packed_rows,
        }
    if not profiles:
        raise RuntimeError(
            f"{backend_mode} arm has no local engine profile to validate",
        )
    return {
        "backend_mode": backend_mode,
        "enabled": backend_mode == "packed_final_noise",
        "profile_exact": True,
        "profiles": profiles,
    }


def _validate_fused_coarse_projector_profiles(
    estep_meta: dict[str, Any],
    *,
    enabled: bool,
    label: str,
) -> dict[str, Any]:
    """Prove that the shared RELION fused projector did or did not execute."""

    from recovar.em.dense_single_volume.helpers.significance import (
        _validate_coarse_selector_audit,
    )

    halfset_profiles = {
        key: value
        for key, value in sorted(estep_meta.items())
        if key.startswith("halfset_") and key.endswith("_profile_summary")
    }
    if not halfset_profiles:
        raise RuntimeError(f"{label} has no halfset execution profile")
    profiles: dict[str, Any] = {}
    for key, value in halfset_profiles.items():
        if not isinstance(value, dict) or "coarse_selector_audit" not in value:
            raise RuntimeError(f"{label} profile {key} lacks coarse_selector_audit")
        try:
            audit = _validate_coarse_selector_audit(value["coarse_selector_audit"])
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"{label} profile {key} has an invalid coarse-selector audit"
            ) from error
        if audit["score_mode"] != "gaussian":
            raise RuntimeError(
                f"{label} profile {key} is not a Gaussian coarse transition"
            )
        if bool(audit["requested_fused"]) != bool(enabled):
            raise RuntimeError(
                f"{label} profile {key} requested_fused="
                f"{audit['requested_fused']!r}, expected {enabled!r}"
            )
        if bool(audit["effective_fused"]) != bool(enabled):
            raise RuntimeError(
                f"{label} profile {key} effective_fused="
                f"{audit['effective_fused']!r}, expected {enabled!r}"
            )
        if audit["requested_workers"] != 0 or audit["effective_workers"] != 0:
            raise RuntimeError(f"{label} changed the fixed zero-worker contract")
        if audit["requested_atomic"] or audit["effective_atomic"]:
            raise RuntimeError(f"{label} changed the fixed non-atomic contract")
        if audit.get("requested_prehalf", False) or audit.get(
            "effective_prehalf",
            False,
        ):
            raise RuntimeError(f"{label} changed the fixed no-prehalf contract")
        profiles[key] = _json_ready(audit)
    if not profiles:
        raise RuntimeError(f"{label} has no Gaussian coarse-selector audit")
    return {
        "enabled": bool(enabled),
        "profile_exact": True,
        "profile_count": len(profiles),
        "profiles": profiles,
    }


def _validate_arm_execution_contract(
    *,
    candidate_mode: str,
    candidate_enabled: bool,
    requested_environment: dict[str, str],
    effective_environment: dict[str, str | None],
    estep_meta: dict[str, Any],
    backend_mode: str | None = None,
) -> dict[str, Any]:
    """Fail closed when a compact arm does not execute the requested profile."""

    if effective_environment != requested_environment:
        raise RuntimeError(
            "same-state arm environment differs from its explicit request: "
            f"requested={requested_environment}, effective={effective_environment}",
        )
    check_compact_profile = _candidate_uses_compact_posterior(candidate_mode)
    check_packed_final_profile = candidate_mode == "packed_final_noise"
    check_fused_coarse_profile = candidate_mode == "fused_coarse_projector"
    contract: dict[str, Any] = {
        "requested_environment": dict(requested_environment),
        "effective_environment": dict(effective_environment),
        "environment_exact": True,
        "profile_checked": (
            check_compact_profile
            or check_packed_final_profile
            or check_fused_coarse_profile
        ),
    }
    if check_packed_final_profile:
        resolved_backend_mode = (
            backend_mode
            if backend_mode is not None
            else candidate_mode if candidate_enabled else "direct"
        )
        contract["packed_final_noise"] = _validate_packed_final_noise_profiles(
            estep_meta,
            backend_mode=resolved_backend_mode,
        )
    if check_fused_coarse_profile:
        contract["fused_coarse_projector"] = (
            _validate_fused_coarse_projector_profiles(
                estep_meta,
                enabled=candidate_enabled,
                label=backend_mode or candidate_mode,
            )
        )
    if not check_compact_profile:
        return contract

    profiles = _coarse_hybrid_profiles(estep_meta)
    contract["hybrid_profiles"] = _json_ready(profiles)
    expected_compact = bool(
        candidate_enabled
        or candidate_mode
        in {
            "all_optimized_stable_shapes",
            "exact_coarse_single_translate",
            "exact_compact_preprocess",
            "fused_pair_fine_score",
        }
    )
    expected_token = "1" if expected_compact else "0"
    if requested_environment[COMPACT_POSTERIOR_ENVIRONMENT] != expected_token:
        raise RuntimeError("compact-posterior request token does not match the arm")
    if not expected_compact:
        if profiles:
            raise RuntimeError(
                "compact-posterior control arm unexpectedly published a hybrid profile",
            )
        contract.update(
            compact_requested=False,
            compact_effective=False,
            packed_deferred_effective=False,
            profile_exact=True,
        )
        return contract

    if not profiles:
        raise RuntimeError(
            "compact-posterior candidate arm did not publish a hybrid profile",
        )
    required_exact = {
        "enabled": True,
        "default_enabled": False,
        "compact_posterior_enabled": True,
        "compact_posterior_default_enabled": False,
        "selected_score_layout": "fixed_capacity_source16",
        "positive_only_scan_role": "correctness_oracle_not_runtime",
        "published_score_source": "exact_relion_source16_or_full_rectangular",
        "expanded_gemm_scores_published": False,
        "whole_batch_fail_closed_fallback": True,
        "score_representation_policy": (
            "compact_only_when_fixed_physical_capacity_is_smaller_than_dense"
        ),
        "fallback_batch_count": 0,
        "fallback_image_count": 0,
    }
    packed_profiles: dict[str, dict[str, Any]] = {}
    for profile_name, profile in profiles.items():
        for field, expected in required_exact.items():
            if profile.get(field) != expected:
                raise RuntimeError(
                    f"{profile_name} compact profile field {field!r} is "
                    f"{profile.get(field)!r}, expected {expected!r}",
                )
        positive_fields = (
            "batch_count",
            "certificate_chunk_count_per_batch",
            "certificate_chunk_rows",
        )
        for field in positive_fields:
            if not isinstance(profile.get(field), int) or profile[field] <= 0:
                raise RuntimeError(
                    f"{profile_name} compact profile field {field!r} must be positive",
                )
        topology_sha256 = profile.get("topology_full_to_compact_sha256")
        if not (
            isinstance(topology_sha256, str)
            and len(topology_sha256) == 64
            and all(character in "0123456789abcdef" for character in topology_sha256)
        ):
            raise RuntimeError(
                f"{profile_name} compact profile has an invalid "
                "topology_full_to_compact_sha256",
            )
        batch_count = profile["batch_count"]
        selected_batch_count = profile.get("selected_rescore_batch_count")
        static_dense_batch_count = profile.get("static_dense_batch_count")
        if not isinstance(selected_batch_count, int) or selected_batch_count < 0:
            raise RuntimeError(
                f"{profile_name} compact profile has invalid "
                "selected_rescore_batch_count",
            )
        if not isinstance(static_dense_batch_count, int) or static_dense_batch_count < 0:
            raise RuntimeError(
                f"{profile_name} compact profile has invalid static_dense_batch_count",
            )
        if selected_batch_count + static_dense_batch_count != batch_count:
            raise RuntimeError(
                f"{profile_name} compact profile did not represent every batch",
            )
        static_fraction = profile.get(
            "static_compact_to_dense_capacity_fraction",
        )
        if not isinstance(static_fraction, (float, int)) or float(static_fraction) <= 0.0:
            raise RuntimeError(
                f"{profile_name} compact static table fraction must be positive",
            )
        prefer_dense = float(static_fraction) >= 1.0
        expected_preference = (
            "dense_full_direct_static_capacity"
            if prefer_dense
            else "compact_selected_exact"
        )
        if profile.get("static_preferred_score_representation") != expected_preference:
            raise RuntimeError(
                f"{profile_name} compact profile has inconsistent static preference",
            )
        expected_representation_counts = {
            expected_preference: batch_count,
        }
        if profile.get("score_representation_batch_counts") != expected_representation_counts:
            raise RuntimeError(
                f"{profile_name} compact profile has inconsistent representation counts",
            )
        if prefer_dense:
            if selected_batch_count != 0 or static_dense_batch_count != batch_count:
                raise RuntimeError(
                    f"{profile_name} compact profile did not choose static dense scoring",
                )
            if profile.get("selected_to_dense_score_table_capacity_fraction") is not None:
                raise RuntimeError(
                    f"{profile_name} static dense profile published a selected table fraction",
                )
        else:
            if selected_batch_count != batch_count or static_dense_batch_count != 0:
                raise RuntimeError(
                    f"{profile_name} compact profile did not select every batch",
                )
            selected_positive_fields = (
                "selected_rescore_image_count",
                "selected_source16_block_count",
                "selected_exact_candidate_count",
                "selected_score_table_capacity_candidates",
                "dense_global_score_table_capacity_candidates",
                "selected_score_table_capacity_bytes_f32",
                "dense_global_score_table_capacity_bytes_f32",
            )
            for field in selected_positive_fields:
                if not isinstance(profile.get(field), int) or profile[field] <= 0:
                    raise RuntimeError(
                        f"{profile_name} compact profile field {field!r} must be positive",
                    )
            table_fraction = profile.get(
                "selected_to_dense_score_table_capacity_fraction",
            )
            if not isinstance(table_fraction, (float, int)) or not 0.0 < float(
                table_fraction,
            ) < 1.0:
                raise RuntimeError(
                    f"{profile_name} compact table fraction must be strictly between 0 and 1",
                )
        if _candidate_uses_packed_deferred(candidate_mode):
            parent_profile = estep_meta[profile_name]
            packed_required = {
                "flat_local_rows_enabled": True,
                "packed_local_projection_enabled": True,
                "defer_packed_vdam_enabled": True,
                "packed_vdam_reuses_flat_score_projection": True,
            }
            packed_profiles[profile_name] = {
                field: parent_profile.get(field) for field in packed_required
            }
            for field, expected in packed_required.items():
                if parent_profile.get(field) != expected:
                    raise RuntimeError(
                        f"{profile_name} packed profile field {field!r} is "
                        f"{parent_profile.get(field)!r}, expected {expected!r}",
                    )
    contract.update(
        compact_requested=True,
        compact_effective=True,
        packed_deferred_effective=_candidate_uses_packed_deferred(candidate_mode),
        profile_exact=True,
    )
    if packed_profiles:
        contract["packed_profiles"] = _json_ready(packed_profiles)
    return contract


def _persistent_cache_snapshot() -> dict[str, Any]:
    """Count persistent JAX programs, including the local-score families."""

    raw_root = os.environ.get("JAX_COMPILATION_CACHE_DIR", "").strip()
    empty_counts = {family: 0 for family in PERSISTENT_CACHE_TARGET_FAMILIES}
    if not raw_root:
        return {
            "available": False,
            "root": None,
            "file_count": 0,
            "bytes": 0,
            "target_family_counts": dict(empty_counts),
            "target_family_bytes": dict(empty_counts),
        }
    root = Path(raw_root)
    family_counts = dict(empty_counts)
    family_bytes = dict(empty_counts)
    file_count = 0
    total_bytes = 0
    if root.is_dir():
        for path in root.iterdir():
            if not path.is_file() or not path.name.endswith("-cache"):
                continue
            try:
                size = int(path.stat().st_size)
            except FileNotFoundError:
                continue
            file_count += 1
            total_bytes += size
            stem = path.name[: -len("-cache")]
            family, separator, digest = stem.rpartition("-")
            if separator and len(digest) == 64 and family in family_counts:
                family_counts[family] += 1
                family_bytes[family] += size
    return {
        "available": True,
        "root": str(root.resolve()),
        "file_count": file_count,
        "bytes": total_bytes,
        "target_family_counts": family_counts,
        "target_family_bytes": family_bytes,
    }


def _persistent_cache_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    counts = {
        family: int(after["target_family_counts"][family])
        - int(before["target_family_counts"][family])
        for family in PERSISTENT_CACHE_TARGET_FAMILIES
    }
    byte_deltas = {
        family: int(after["target_family_bytes"][family])
        - int(before["target_family_bytes"][family])
        for family in PERSISTENT_CACHE_TARGET_FAMILIES
    }
    return {
        "available": bool(before["available"] and after["available"]),
        "file_count_delta": int(after["file_count"]) - int(before["file_count"]),
        "bytes_delta": int(after["bytes"]) - int(before["bytes"]),
        "target_family_count_delta": counts,
        "target_family_bytes_delta": byte_deltas,
        "no_new_target_family_programs": all(value == 0 for value in counts.values()),
    }


LOCAL_TIMING_FIELDS = (
    "em_time_s",
    "accounted_em_time_s",
    "unattributed_em_time_s",
    "preprocess_time_s",
    "batch_fetch_time_s",
    "bucket_build_time_s",
    "projection_time_s",
    "big_jit_bucket_s",
    "local_score_s",
    "local_normalize_s",
    "local_significance_s",
    "local_pack_s",
    "local_noise_s",
    "local_mstep_s",
    "local_postprocess_s",
    "local_final_accumulator_s",
    "transfer_total_to_host_s",
)


def _local_performance_metric_name(field: str) -> str:
    """Return the canonical arm-summary name for one local-engine timer."""

    return field if field.startswith("local_") else f"local_{field}"


def _local_timing_summary(estep_meta: dict[str, Any]) -> dict[str, Any]:
    by_profile: dict[str, dict[str, float]] = {}
    totals = {field: 0.0 for field in LOCAL_TIMING_FIELDS}
    for name, profile in sorted(estep_meta.items()):
        if not isinstance(profile, dict) or "em_time_s" not in profile:
            continue
        missing = [field for field in LOCAL_TIMING_FIELDS if field not in profile]
        if missing:
            raise RuntimeError(
                f"local timing profile {name} omitted canonical fields {missing}",
            )
        timings = {field: float(profile[field]) for field in LOCAL_TIMING_FIELDS}
        invalid = [
            field
            for field, value in timings.items()
            if not np.isfinite(value) or value < 0.0
        ]
        if invalid:
            raise RuntimeError(
                f"local timing profile {name} has invalid fields {invalid}",
            )
        by_profile[name] = timings
        for field, value in timings.items():
            totals[field] += value
    return {
        "profile_count": len(by_profile),
        "totals_s": totals,
        "by_profile_s": by_profile,
    }


def _arm_performance_summary(
    *,
    wall_s: float,
    estep_meta: dict[str, Any],
    persistent_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sparse = estep_meta.get("sparse_pass2_profile_summary", {})
    if not isinstance(sparse, dict):
        raise RuntimeError("sparse pass-2 profile summary is not a mapping")
    summary: dict[str, Any] = {"wall_s": float(wall_s)}
    for field in (
        "pass1_time_s",
        "pass2_time_s",
        "max_significant_samples",
        "mean_significant_samples",
    ):
        if field in sparse:
            summary[field] = _json_ready(sparse[field])
    local_timing = _local_timing_summary(estep_meta)
    summary["local_timing"] = local_timing
    if local_timing["profile_count"]:
        for field, value in local_timing["totals_s"].items():
            summary[_local_performance_metric_name(field)] = value
    if persistent_cache is not None:
        summary["persistent_cache"] = persistent_cache
    table_fields = (
        "score_representation_policy",
        "static_preferred_score_representation",
        "score_representation_batch_counts",
        "selected_rescore_batch_count",
        "static_dense_batch_count",
        "fallback_batch_count",
        "selected_rescore_image_count",
        "static_dense_image_count",
        "overflow_latch_scope",
        "overflow_latch_active_at_return",
        "overflow_latch_activation_count",
        "overflow_latch_static_dense_batch_count",
        "overflow_latch_static_dense_image_count",
        "selected_source16_block_count",
        "selected_exact_candidate_count",
        "full_candidate_count_for_selected_images",
        "selected_exact_candidate_fraction",
        "selected_score_table_capacity_candidates",
        "dense_global_score_table_capacity_candidates",
        "selected_score_table_capacity_bytes_f32",
        "dense_global_score_table_capacity_bytes_f32",
        "selected_to_dense_score_table_capacity_fraction",
        "static_compact_to_dense_capacity_fraction",
        "max_selected_blocks_per_image",
        "selected_block_capacity",
    )
    profiles = _coarse_hybrid_profiles(estep_meta)
    if profiles:
        summary["coarse_hybrid_tables"] = {
            name: {
                field: _json_ready(profile[field])
                for field in table_fields
                if field in profile
            }
            for name, profile in profiles.items()
        }
    return summary


def _incremental_backend_for_label(label: str) -> str:
    if label == "direct_oracle":
        return "direct"
    if "packed_final_noise" in label:
        return "packed_final_noise"
    if "packed_deferred" in label:
        return "packed_deferred"
    raise ValueError(f"unsupported incremental arm label: {label}")


def _incremental_arm_specs() -> tuple[tuple[str, str], ...]:
    return tuple(
        (label, _incremental_backend_for_label(label))
        for label in PACKED_FINAL_NOISE_INCREMENTAL_GATE_ARM_ORDER
    )


def _incremental_pair_labels() -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for panel in (
        PACKED_FINAL_NOISE_INCREMENTAL_ABBA_ARM_ORDER,
        PACKED_FINAL_NOISE_INCREMENTAL_BAAB_ARM_ORDER,
    ):
        baseline = [
            label
            for label in panel
            if _incremental_backend_for_label(label) == "packed_deferred"
        ]
        candidate = [
            label
            for label in panel
            if _incremental_backend_for_label(label) == "packed_final_noise"
        ]
        pairs.extend(
            [
                (baseline[0], baseline[1]),
                (candidate[0], candidate[1]),
                *((left, right) for left in baseline for right in candidate),
            ]
        )
    pairs.extend(
        [
            (
                PACKED_FINAL_NOISE_INCREMENTAL_ABBA_ARM_ORDER[0],
                PACKED_FINAL_NOISE_INCREMENTAL_BAAB_ARM_ORDER[1],
            ),
            (
                PACKED_FINAL_NOISE_INCREMENTAL_ABBA_ARM_ORDER[1],
                PACKED_FINAL_NOISE_INCREMENTAL_BAAB_ARM_ORDER[0],
            ),
            (
                "direct_oracle",
                PACKED_FINAL_NOISE_INCREMENTAL_ABBA_ARM_ORDER[0],
            ),
            (
                "direct_oracle",
                PACKED_FINAL_NOISE_INCREMENTAL_ABBA_ARM_ORDER[1],
            ),
        ]
    )
    return tuple(pairs)


def _hybrid_image_batch_request_for_label(label: str) -> int | None:
    if label == "direct_oracle":
        return None
    if "batch110" in label:
        return HYBRID_IMAGE_BATCH_CONTROL_REQUEST
    if "batch200" in label:
        return HYBRID_IMAGE_BATCH_CANDIDATE_REQUEST
    raise ValueError(f"unsupported hybrid image-batch arm label: {label}")


def _hybrid_image_batch_arm_specs() -> tuple[tuple[str, int | None], ...]:
    return tuple(
        (label, _hybrid_image_batch_request_for_label(label))
        for label in HYBRID_IMAGE_BATCH_GATE_ARM_ORDER
    )


def _hybrid_image_batch_pair_labels() -> tuple[tuple[str, str], ...]:
    """Compare every pair in the oracle plus mirrored eight-arm panel."""

    return tuple(
        (left, right)
        for left_index, left in enumerate(HYBRID_IMAGE_BATCH_GATE_ARM_ORDER)
        for right in HYBRID_IMAGE_BATCH_GATE_ARM_ORDER[left_index + 1 :]
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_manifest(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def _native_checkpoint_file_manifests(
    continuation,
    *,
    data_dir: Path,
) -> dict[str, Any]:
    """Hash every STAR, reference, and gradient moment consumed by restart."""

    import starfile

    import recovar.em.initial_model.driver as driver
    from recovar.data_io.starfile import read_star

    files: dict[str, Path] = {
        "optimiser_star": continuation.optimiser_star,
        "model_star": continuation.model_star,
        "data_star": continuation.data_star,
        "sampling_star": continuation.sampling_star,
    }
    model = starfile.read(continuation.model_star, always_dict=True)
    classes = model.get("model_classes")
    if classes is None or len(classes) != int(continuation.state.K):
        raise RuntimeError("native checkpoint model class table changed after load")
    for class_offset, (_row_index, row) in enumerate(classes.iterrows()):
        class_number = class_offset + 1
        reference = driver._resolve_relion_checkpoint_path(
            str(row["rlnReferenceImage"]),
            owner=continuation.model_star,
        )
        moment1 = driver._resolve_relion_checkpoint_path(
            str(row["rlnGradMoment1"]),
            owner=continuation.model_star,
        )
        moment2 = driver._resolve_relion_checkpoint_path(
            str(row["rlnGradMoment2"]),
            owner=continuation.model_star,
        )
        files[f"class_{class_number:03d}_reference"] = reference
        files[f"class_{class_number:03d}_grad_moment1_half1"] = moment1
        files[f"class_{class_number:03d}_grad_moment1_half2"] = (
            driver._second_pseudo_half_moment_path(
                moment1,
                nr_classes=int(continuation.state.K),
            )
        )
        files[f"class_{class_number:03d}_grad_moment2"] = moment2
    main_star, _optics_star = read_star(str(continuation.data_star))
    image_column = next(
        (
            name
            for name in ("_rlnImageName", "rlnImageName")
            if name in main_star.columns
        ),
        None,
    )
    if image_column is None:
        raise RuntimeError("native checkpoint data STAR lacks _rlnImageName")
    stack_paths: set[Path] = set()
    for image_name in main_star[image_column].astype(str).to_numpy():
        if "@" not in image_name:
            raise RuntimeError(
                f"native checkpoint image name lacks stack index: {image_name!r}"
            )
        stack_token = image_name.split("@", 1)[1]
        stack_path = Path(stack_token).expanduser()
        if not stack_path.is_absolute():
            stack_path = data_dir / stack_path
        stack_paths.add(stack_path.resolve(strict=True))
    if not stack_paths:
        raise RuntimeError("native checkpoint data STAR has no particle stacks")
    for stack_number, stack_path in enumerate(sorted(stack_paths), start=1):
        files[f"particle_stack_{stack_number:03d}"] = stack_path
    return {name: _file_manifest(path) for name, path in sorted(files.items())}


def _resolve_target_particle_index(main_star, image_name: str) -> int:
    """Resolve one checkpoint-local particle row by exact RELION image name."""

    image_name = str(image_name).strip()
    if not image_name:
        raise ValueError("target-image-name must be non-empty")
    column_name = next(
        (
            name
            for name in ("_rlnImageName", "rlnImageName")
            if name in main_star.columns
        ),
        None,
    )
    if column_name is None:
        raise ValueError("target checkpoint STAR lacks _rlnImageName")
    names = np.asarray(main_star[column_name].astype(str).to_numpy(), dtype=str)
    matches = np.flatnonzero(names == image_name)
    if matches.size != 1:
        raise ValueError(
            f"target image {image_name!r} must match exactly one checkpoint row; "
            f"found {int(matches.size)}"
        )
    return int(matches[0])


def _target_star_metadata(main_star, row_index: int) -> dict[str, Any]:
    """Return a small source-STAR snapshot for one checkpoint-local row."""

    row_index = int(row_index)
    if row_index < 0 or row_index >= len(main_star):
        raise IndexError("target checkpoint row is outside the STAR table")
    row = main_star.iloc[row_index]

    def _value(*names: str, cast=None):
        for name in names:
            if name in main_star.columns:
                value = row[name]
                return cast(value) if cast is not None else value
        return None

    eulers = [
        _value("_rlnAngleRot", "rlnAngleRot", cast=float),
        _value("_rlnAngleTilt", "rlnAngleTilt", cast=float),
        _value("_rlnAnglePsi", "rlnAnglePsi", cast=float),
    ]
    origins_angstrom = [
        _value("_rlnOriginXAngst", "rlnOriginXAngst", cast=float),
        _value("_rlnOriginYAngst", "rlnOriginYAngst", cast=float),
    ]
    return {
        "checkpoint_row_index": row_index,
        "image_name": _value("_rlnImageName", "rlnImageName", cast=str),
        "euler_degrees": eulers if all(value is not None for value in eulers) else None,
        "origin_angstrom": (
            origins_angstrom
            if all(value is not None for value in origins_angstrom)
            else None
        ),
        "class_number": _value("_rlnClassNumber", "rlnClassNumber", cast=int),
        "max_posterior": _value(
            "_rlnMaxValueProbDistribution",
            "rlnMaxValueProbDistribution",
            cast=float,
        ),
        "significant_count": _value(
            "_rlnNrOfSignificantSamples",
            "rlnNrOfSignificantSamples",
            cast=int,
        ),
    }


def _target_particle_summary(
    particle_state,
    estep_meta: dict[str, Any] | None,
    *,
    row_index: int,
    pixel_size: float,
) -> dict[str, Any]:
    """Summarize one particle's hard decision after a transition."""

    from recovar.utils.helpers import R_to_relion

    row_index = int(row_index)
    n_particles = int(np.asarray(particle_state.translation_offsets).shape[0])
    if row_index < 0 or row_index >= n_particles:
        raise IndexError("target particle row is outside the particle state")
    visited = bool(
        np.asarray(particle_state.visited, dtype=bool)[row_index]
        if particle_state.visited is not None
        else float(np.asarray(particle_state.max_posterior)[row_index]) > 0.0
    )

    def _optional_row(name: str):
        value = getattr(particle_state, name)
        if value is None:
            return None
        return np.asarray(value)[row_index]

    rotation = _optional_row("best_pose_rotations")
    eulers = (
        None
        if rotation is None
        else np.asarray(
            R_to_relion(np.asarray(rotation)[None, ...], degrees=True),
            dtype=np.float64,
        )[0].tolist()
    )
    selected_position = None
    significant_count = None
    if estep_meta is not None and estep_meta.get("selected_particle_ids") is not None:
        selected_ids = np.asarray(
            estep_meta["selected_particle_ids"],
            dtype=np.int64,
        ).reshape(-1)
        positions = np.flatnonzero(selected_ids == row_index)
        if positions.size > 1:
            raise RuntimeError("target particle appeared more than once in the E-step subset")
        if positions.size == 1:
            selected_position = int(positions[0])
            counts = estep_meta.get("significant_counts")
            if counts is not None:
                counts = np.asarray(counts).reshape(-1)
                if counts.shape != selected_ids.shape:
                    raise RuntimeError(
                        "significant_counts does not align with selected_particle_ids"
                    )
                significant_count = int(counts[selected_position])

    origin_px = np.asarray(particle_state.translation_offsets, dtype=np.float64)[
        row_index
    ]
    best_translation = _optional_row("best_pose_translations")
    rotation_id = _optional_row("best_pose_rotation_ids")
    pose_assignment = _optional_row("pose_assignments")
    class_index = int(np.asarray(particle_state.class_assignments)[row_index])
    summary = {
        "checkpoint_row_index": row_index,
        "selected_in_transition": selected_position is not None,
        "selected_position": selected_position,
        "visited": visited,
        "class_index_zero_based": class_index,
        "class_number": class_index + 1 if visited else 0,
        "pose_assignment": None if pose_assignment is None else int(pose_assignment),
        "best_pose_rotation_id": None if rotation_id is None else int(rotation_id),
        "best_pose_translation_px": (
            None
            if best_translation is None
            else np.asarray(best_translation, dtype=np.float64).tolist()
        ),
        "origin_offset_px": origin_px.tolist(),
        "origin_offset_angstrom": (origin_px * float(pixel_size)).tolist(),
        "euler_degrees": eulers,
        "max_posterior": float(np.asarray(particle_state.max_posterior)[row_index]),
        "significant_count": significant_count,
    }
    hard_state = {
        key: summary[key]
        for key in (
            "visited",
            "class_number",
            "pose_assignment",
            "best_pose_rotation_id",
            "best_pose_translation_px",
            "origin_offset_px",
            "euler_degrees",
            "max_posterior",
            "significant_count",
        )
    }
    summary["hard_state_sha256"] = _sha256_bytes(
        json.dumps(hard_state, sort_keys=True, separators=(",", ":")).encode()
    )
    return summary


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


@contextmanager
def _temporary_environment(values: dict[str, str | None]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _array_manifest(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    result: dict[str, Any] = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _array_sha256(array),
        "size": int(array.size),
    }
    if array.size and array.dtype.kind in "biufc":
        absolute = np.abs(array)
        result.update(
            finite=bool(np.all(np.isfinite(array))),
            max_abs=float(np.max(absolute)),
            l2_norm=float(np.linalg.norm(array.reshape(-1))),
        )
    return result


def _array_comparison(left: Any, right: Any) -> dict[str, Any]:
    lhs = np.asarray(left)
    rhs = np.asarray(right)
    result: dict[str, Any] = {
        "left_dtype": lhs.dtype.str,
        "right_dtype": rhs.dtype.str,
        "left_shape": list(lhs.shape),
        "right_shape": list(rhs.shape),
        "comparable": bool(lhs.shape == rhs.shape),
    }
    if lhs.shape != rhs.shape:
        result["exact_equal"] = False
        return result
    exact = np.array_equal(lhs, rhs, equal_nan=True)
    result["exact_equal"] = bool(exact)
    unequal = ~(lhs == rhs)
    if lhs.dtype.kind in "fc" or rhs.dtype.kind in "fc":
        unequal &= ~(np.isnan(lhs) & np.isnan(rhs))
    result["mismatch_count"] = int(np.count_nonzero(unequal))
    if np.any(unequal):
        result["first_mismatch_flat_indices"] = np.flatnonzero(unequal.reshape(-1))[:32].tolist()
    if lhs.dtype.kind in "biufc" and rhs.dtype.kind in "biufc" and lhs.size:
        delta = lhs.astype(np.complex128 if (lhs.dtype.kind == "c" or rhs.dtype.kind == "c") else np.float64) - rhs
        absolute = np.abs(delta)
        scale = max(
            float(np.linalg.norm(lhs.reshape(-1))),
            float(np.linalg.norm(rhs.reshape(-1))),
            float(np.finfo(np.float64).tiny),
        )
        result.update(
            max_abs_delta=float(np.max(absolute)),
            normalized_l2_delta=float(np.linalg.norm(delta.reshape(-1)) / scale),
        )
        if not np.iscomplexobj(delta):
            result["signed_mean_delta"] = float(np.mean(delta, dtype=np.float64))
    return result


def _dataclass_manifest(value: Any) -> dict[str, Any]:
    if not dataclasses.is_dataclass(value):
        raise TypeError("value must be a dataclass instance")
    fields: dict[str, Any] = {}
    for field in dataclasses.fields(value):
        item = getattr(value, field.name)
        fields[field.name] = (
            {"kind": "none"}
            if item is None
            else {"kind": "array", **_array_manifest(item)}
            if isinstance(item, np.ndarray)
            else {"kind": "scalar", "value": _json_ready(item)}
        )
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    return {"fields": fields, "manifest_sha256": _sha256_bytes(encoded)}


def _dataclass_comparison(left: Any, right: Any) -> dict[str, Any]:
    if type(left) is not type(right) or not dataclasses.is_dataclass(left):
        raise TypeError("values must be matching dataclass instances")
    result: dict[str, Any] = {}
    for field in dataclasses.fields(left):
        lhs = getattr(left, field.name)
        rhs = getattr(right, field.name)
        if lhs is None or rhs is None:
            result[field.name] = {"exact_equal": lhs is None and rhs is None}
        elif isinstance(lhs, np.ndarray) or isinstance(rhs, np.ndarray):
            result[field.name] = _array_comparison(lhs, rhs)
        else:
            result[field.name] = {"exact_equal": bool(lhs == rhs), "left": _json_ready(lhs), "right": _json_ready(rhs)}
    return result


def _accumulator_manifest(accumulators: list[Any]) -> list[dict[str, Any]]:
    return [_dataclass_manifest(value) for value in accumulators]


def _accumulator_comparison(left: list[Any], right: list[Any]) -> dict[str, Any]:
    if len(left) != len(right):
        return {"comparable": False, "left_count": len(left), "right_count": len(right)}
    return {
        "comparable": True,
        "entries": [
            _dataclass_comparison(lhs, rhs) for lhs, rhs in zip(left, right)
        ],
    }


def _meta_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in META_ARRAY_KEYS:
        if key in left or key in right:
            result[key] = (
                _array_comparison(left[key], right[key])
                if key in left and key in right
                else {"exact_equal": False, "left_present": key in left, "right_present": key in right}
            )
    left_ids = np.asarray(left.get("selected_particle_ids", []), dtype=np.int64)
    right_ids = np.asarray(right.get("selected_particle_ids", []), dtype=np.int64)
    if np.array_equal(left_ids, right_ids):
        for key in ("best_pose_rotation_ids", "pose_assignments", "significant_counts"):
            if key not in left or key not in right:
                continue
            lhs = np.asarray(left[key])
            rhs = np.asarray(right[key])
            if lhs.shape == rhs.shape == left_ids.shape:
                mask = lhs != rhs
                result[key]["mismatching_selected_particle_ids"] = left_ids[mask].tolist()
    return result


def _support_audits(meta: dict[str, Any]) -> dict[str, Any]:
    """Return every support audit, including audits nested in profile summaries."""
    result: dict[str, Any] = {}
    for key, value in meta.items():
        if "coarse_significance_support_audit" in key:
            result[key] = _json_ready(value)
        if isinstance(value, dict) and "coarse_significance_support_audit" in value:
            result[f"{key}.coarse_significance_support_audit"] = _json_ready(
                value["coarse_significance_support_audit"]
            )
    return result


def _support_audit_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    lhs_audits = _support_audits(left)
    rhs_audits = _support_audits(right)
    entries: dict[str, Any] = {}
    for key in sorted(set(lhs_audits) | set(rhs_audits)):
        lhs = lhs_audits.get(key)
        rhs = rhs_audits.get(key)
        entries[key] = {
            "exact_equal": lhs == rhs,
            "left_present": key in lhs_audits,
            "right_present": key in rhs_audits,
            "left_sha256": (
                _sha256_bytes(json.dumps(lhs, sort_keys=True, separators=(",", ":")).encode())
                if key in lhs_audits
                else None
            ),
            "right_sha256": (
                _sha256_bytes(json.dumps(rhs, sort_keys=True, separators=(",", ":")).encode())
                if key in rhs_audits
                else None
            ),
            "left_aggregate_support_sha256": (
                lhs.get("aggregate_support_sha256") if isinstance(lhs, dict) else None
            ),
            "right_aggregate_support_sha256": (
                rhs.get("aggregate_support_sha256") if isinstance(rhs, dict) else None
            ),
        }
    return {
        "exact_equal": bool(entries) and all(entry["exact_equal"] for entry in entries.values()),
        "left_count": len(lhs_audits),
        "right_count": len(rhs_audits),
        "entries": entries,
    }


def _capture_direct_checkpoint(
    *,
    fixture_dir: Path,
    acceptance: dict[str, Any],
    output_root: Path,
    checkpoint_iteration: int,
    image_batch_size: int,
    checkpoint_candidate_mode: str = "hybrid",
    checkpoint_candidate_enabled: bool = False,
    native_checkpoint_optimiser: Path | None = None,
    native_checkpoint_data_star: Path | None = None,
    native_data_dir: Path | None = None,
) -> dict[str, Any]:
    import recovar.em.initial_model.driver as driver
    from scripts import run_ab_initio
    from scripts.run_vdam_relion_parity_case import build_recovar_command

    native_mode = native_checkpoint_optimiser is not None
    if native_mode != (
        native_checkpoint_data_star is not None and native_data_dir is not None
    ):
        raise ValueError("native checkpoint capture requires all native inputs")
    input_star = (
        native_checkpoint_data_star
        if native_mode
        else fixture_dir / "particles.star"
    )
    command_data_dir = native_data_dir if native_mode else fixture_dir
    assert input_star is not None and command_data_dir is not None
    definition = acceptance["science_contract"]["definition"]
    command = build_recovar_command(
        input_star=input_star,
        output_prefix=output_root / "checkpoint" / "run",
        fixture_dir=command_data_dir,
        definition=definition,
        image_batch_size=image_batch_size,
    )
    argv = list(command[3:])
    write_index = argv.index("--grad_write_iter") + 1
    argv[write_index] = str(checkpoint_iteration)
    if native_mode:
        argv.extend(
            (
                "--diagnostic_continue_optimiser",
                str(native_checkpoint_optimiser),
                "--diagnostic_stop_after_iteration",
                str(checkpoint_iteration + 1),
                "--no_iter_artifacts",
            )
        )
    else:
        argv.extend(("--diagnostic_stop_after_iteration", str(checkpoint_iteration)))
    if (
        checkpoint_candidate_mode == "all_optimized_stable_shapes"
        and checkpoint_candidate_enabled
    ):
        argv.append("--stable-fourier-window-shapes")

    captured: dict[str, Any] = {
        "argv": argv,
        "output_root": output_root,
        "checkpoint_source": "native_relion" if native_mode else "recovar_direct",
    }
    original_expectation_factory = driver._native_expectation_step
    original_run = driver.run_native_initial_model
    original_continuation_loader = driver._load_native_vdam_continuation
    original_iteration_loop = driver.run_vdam_iterations

    def capture_expectation_factory(dataset, opts, noise_variance, particle_state, sampling_state=None, optics_state=None):
        captured.update(
            dataset=dataset,
            opts=opts,
            noise_variance=noise_variance,
            particle_state=particle_state,
            sampling_state=sampling_state,
            optics_state=optics_state,
        )
        return original_expectation_factory(
            dataset,
            opts,
            noise_variance,
            particle_state,
            sampling_state,
            optics_state,
        )

    def capture_run(opts):
        result = original_run(opts)
        captured["result"] = result
        return result

    def capture_continuation(*args, **kwargs):
        continuation = original_continuation_loader(*args, **kwargs)
        captured["continuation"] = continuation
        return continuation

    def capture_loaded_state(state, *positional, **kwargs):
        if positional:
            raise RuntimeError("native checkpoint loader received unexpected positional args")
        captured["load_only_iteration_call"] = {
            "start_iteration": int(kwargs["start_iteration"]),
            "diagnostic_stop_after_iteration": int(
                kwargs["diagnostic_stop_after_iteration"]
            ),
        }
        return state

    checkpoint_environment = {
        **_candidate_environment(
            checkpoint_candidate_mode,
            enabled=checkpoint_candidate_enabled,
        ),
        "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT": "0",
        "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS": "0",
    }
    driver._native_expectation_step = capture_expectation_factory
    driver.run_native_initial_model = capture_run
    if native_mode:
        driver._load_native_vdam_continuation = capture_continuation
        driver.run_vdam_iterations = capture_loaded_state
    try:
        with _temporary_environment(
            checkpoint_environment
        ):
            captured["effective_environment"] = {
                name: os.environ.get(name) for name in checkpoint_environment
            }
            status = int(run_ab_initio.main(argv))
    finally:
        driver._native_expectation_step = original_expectation_factory
        driver.run_native_initial_model = original_run
        driver._load_native_vdam_continuation = original_continuation_loader
        driver.run_vdam_iterations = original_iteration_loop
    if status != 0:
        raise RuntimeError(f"direct checkpoint trajectory exited with status {status}")
    required = {
        "dataset",
        "opts",
        "noise_variance",
        "particle_state",
        "sampling_state",
        "optics_state",
        "result",
    }
    missing = required - set(captured)
    if missing:
        raise RuntimeError(f"direct checkpoint capture is incomplete: {sorted(missing)}")
    if int(captured["result"].state.iter) != checkpoint_iteration:
        raise RuntimeError("direct checkpoint stopped at the wrong iteration")
    if captured["effective_environment"] != checkpoint_environment:
        raise RuntimeError("checkpoint trajectory environment was not exact")
    expected_stable = bool(
        not native_mode
        and checkpoint_candidate_mode == "all_optimized_stable_shapes"
        and checkpoint_candidate_enabled
    )
    if bool(captured["opts"].stable_fourier_window_shapes) != expected_stable:
        raise RuntimeError("checkpoint trajectory stable-Fourier option was not exact")
    captured["checkpoint_execution_contract"] = {
        "source": captured["checkpoint_source"],
        "requested_environment": checkpoint_environment,
        "effective_environment": captured["effective_environment"],
        "environment_exact": True,
        "stable_fourier_window_shapes": expected_stable,
        "transition_executed_during_capture": not native_mode,
    }
    if native_mode:
        required_native = {"continuation", "load_only_iteration_call"}
        missing_native = required_native - set(captured)
        if missing_native:
            raise RuntimeError(
                "native checkpoint capture is incomplete: "
                f"{sorted(missing_native)}"
            )
        continuation = captured["continuation"]
        if int(continuation.iteration) != checkpoint_iteration:
            raise RuntimeError("native continuation has the wrong iteration")
        expected_call = {
            "start_iteration": checkpoint_iteration,
            "diagnostic_stop_after_iteration": checkpoint_iteration + 1,
        }
        if captured["load_only_iteration_call"] != expected_call:
            raise RuntimeError("native checkpoint load-only call contract changed")
        captured["checkpoint_execution_contract"].update(
            {
                "load_only_call": captured["load_only_iteration_call"],
                "input_files": _native_checkpoint_file_manifests(
                    continuation,
                    data_dir=native_data_dir,
                ),
                "data_dir": str(native_data_dir.resolve(strict=True)),
            }
        )
    captured["expectation_factory"] = original_expectation_factory
    return captured


def _run_transition_arm(
    checkpoint: dict[str, Any],
    *,
    label: str,
    candidate_mode: str,
    candidate_enabled: bool,
    checkpoint_iteration: int,
    backend_mode: str | None = None,
    hybrid_image_batch_request: int | None = None,
    exact_coarse_profile_enabled: bool = False,
    fused_posterior_dump_original_index: int | None = None,
    coarse_prefix_dump_original_index: int | None = None,
) -> dict[str, Any]:
    import recovar.em.initial_model.driver as driver
    from recovar.data_io.starfile import read_star
    from recovar.em.initial_model.schedules import default_subset_sizes_for_3d_initial_model

    if hybrid_image_batch_request is not None and not (
        candidate_mode == "all_optimized" and candidate_enabled and backend_mode is None
    ):
        raise ValueError(
            "hybrid image-batch arms require the complete all-optimized backend",
        )

    state = copy.deepcopy(checkpoint["result"].state)
    particle_state = copy.deepcopy(checkpoint["particle_state"])
    sampling_state = copy.deepcopy(checkpoint["sampling_state"])
    initial_state_manifest = _dataclass_manifest(state)
    initial_particle_state_manifest = _dataclass_manifest(particle_state)
    initial_sampling_state_manifest = _dataclass_manifest(sampling_state)
    opts = checkpoint["opts"]
    if candidate_mode in {
        "stable_shapes",
        "all_optimized",
        "all_optimized_stable_shapes",
        "exact_coarse_single_translate",
        "exact_compact_preprocess",
        "fused_pair_fine_score",
    }:
        opts = dataclasses.replace(
            opts,
            stable_fourier_window_shapes=bool(
                candidate_enabled
                or candidate_mode
                in {
                    "exact_coarse_single_translate",
                    "exact_compact_preprocess",
                    "fused_pair_fine_score",
                }
            ),
        )
    dataset = checkpoint["dataset"]
    expectation_step = checkpoint["expectation_factory"](
        dataset,
        opts,
        checkpoint["noise_variance"],
        particle_state,
        sampling_state,
        checkpoint["optics_state"],
    )
    captured: dict[str, Any] = {}

    def capture_expectation(current, particle_ids, halfset_ids):
        accumulators, meta = expectation_step(current, particle_ids, halfset_ids)
        captured["accumulators"] = accumulators
        captured["estep_meta"] = copy.deepcopy(meta)
        return accumulators, meta

    def capture_artifact(_current, _iteration, meta):
        captured["post_iteration_meta"] = copy.deepcopy(meta)
        driver._record_native_sampling_post_iteration(
            sampling_state,
            _current,
            iteration=int(_iteration),
            meta=meta,
        )

    post_mstep_update = None
    if opts.do_solvent:
        solvent_mask = driver.relion_solvent_mask(
            ori_size=int(state.ori_size),
            pixel_size=float(state.pixel_size),
            particle_diameter_ang=float(opts.particle_diameter),
            width_mask_edge_px=float(opts.width_mask_edge_px),
        )

        def post_mstep_update(current, iteration, meta):
            current = driver.relion_solvent_flatten_state(current, mask=solvent_mask)
            return driver._maybe_replay_iteration_references(current, iteration=iteration, meta=meta)

    main_star, _optics_star = read_star(opts.fn_img)
    optics_group_by_particle = driver._optics_group_indices(main_star)
    particle_order = driver._micrograph_sort_order(main_star)
    grad_ini_subset_size, grad_fin_subset_size = default_subset_sizes_for_3d_initial_model(
        int(dataset.n_images)
    )
    if backend_mode is not None:
        resolved_backend_mode = backend_mode
    elif candidate_mode == "exact_coarse_single_translate":
        resolved_backend_mode = (
            "all_optimized_exact_coarse_single_translate_on"
            if candidate_enabled
            else "all_optimized_exact_coarse_single_translate_off"
        )
    elif candidate_mode == "exact_compact_preprocess":
        resolved_backend_mode = (
            "all_optimized_exact_compact_preprocess_on"
            if candidate_enabled
            else "all_optimized_exact_compact_preprocess_off"
        )
    elif candidate_mode == "fused_pair_fine_score":
        resolved_backend_mode = (
            "all_optimized_fused_pair_fine_score_on"
            if candidate_enabled
            else "all_optimized_fused_pair_fine_score_off"
        )
    else:
        resolved_backend_mode = candidate_mode if candidate_enabled else "direct"
    if backend_mode is None:
        requested_environment = _candidate_environment(
            candidate_mode,
            enabled=candidate_enabled,
        )
    elif resolved_backend_mode == "direct":
        requested_environment = _candidate_environment(candidate_mode, enabled=False)
    else:
        requested_environment = _candidate_environment(
            resolved_backend_mode,
            enabled=True,
        )
    if hybrid_image_batch_request is not None:
        requested_environment[HYBRID_IMAGE_BATCH_ENVIRONMENT] = str(
            hybrid_image_batch_request,
        )
    if candidate_mode in {
        "exact_coarse_single_translate",
        "exact_compact_preprocess",
    }:
        requested_environment[EXACT_COARSE_ASSEMBLY_PROFILE_ENVIRONMENT] = (
            "1" if exact_coarse_profile_enabled else "0"
        )
    effective_environment: dict[str, str | None] = {}
    diagnostic_environment: dict[str, str] = {}
    if fused_posterior_dump_original_index is not None:
        diagnostic_environment = {
            "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR": str(
                checkpoint["output_root"] / "fused_posterior_dumps"
            ),
            "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_GLOBAL_INDICES": str(
                int(fused_posterior_dump_original_index)
            ),
            "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_ITERATION": str(
                int(checkpoint_iteration) + 1
            ),
            "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_LABEL": label,
            "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_SCORES": "1",
        }
    if coarse_prefix_dump_original_index is not None:
        diagnostic_environment.update(
            {
                "RECOVAR_COARSE_RUNTIME_PREFIX_DUMP_DIR": str(
                    checkpoint["output_root"] / "coarse_prefix_dumps"
                ),
                "RECOVAR_COARSE_RUNTIME_PREFIX_DUMP_ORIGINAL_INDICES": str(
                    int(coarse_prefix_dump_original_index)
                ),
                "RECOVAR_COARSE_RUNTIME_PREFIX_DUMP_LABEL": label,
            },
        )
    persistent_cache_before = _persistent_cache_snapshot()
    wall_clock_started_epoch_s = time.time()
    started = time.perf_counter()
    with _temporary_environment(
        {
            **requested_environment,
            **diagnostic_environment,
            "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT": "1",
            "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS": "1",
        }
    ):
        effective_environment = {
            name: os.environ.get(name) for name in requested_environment
        }
        final_state = driver.run_vdam_iterations(
            state,
            nr_particles=int(dataset.n_images),
            optics_group_by_particle=optics_group_by_particle,
            grad_ini_subset_size=grad_ini_subset_size,
            grad_fin_subset_size=grad_fin_subset_size,
            tau2_fudge_arg=float(opts.tau2_fudge),
            grad_em_iters=int(opts.grad_em_iters),
            random_seed=int(opts.random_seed),
            rnd_unif_factory=driver._relion_rnd_unif_factory,
            expectation_step=capture_expectation,
            iter_artifact_sink=capture_artifact,
            post_mstep_update=post_mstep_update,
            particle_order=particle_order,
            grad_ini_frac=float(opts.grad_ini_frac),
            grad_fin_frac=float(opts.grad_fin_frac),
            grad_stepsize=float(opts.stepsize),
            mu=float(opts.mu),
            projector_padding_factor=int(opts.padding_factor),
            start_iteration=checkpoint_iteration,
            diagnostic_stop_after_iteration=checkpoint_iteration + 1,
        )
    wall_s = float(time.perf_counter() - started)
    wall_clock_ended_epoch_s = time.time()
    persistent_cache_after = _persistent_cache_snapshot()
    persistent_cache = {
        "before": persistent_cache_before,
        "after": persistent_cache_after,
        "delta": _persistent_cache_delta(
            persistent_cache_before,
            persistent_cache_after,
        ),
    }
    if int(final_state.iter) != checkpoint_iteration + 1:
        raise RuntimeError(f"{label} stopped at the wrong iteration")
    if "accumulators" not in captured or "estep_meta" not in captured:
        raise RuntimeError(f"{label} did not capture its E-step boundary")
    stable_fourier_window_contract = None
    stable_flat_capacity_contract = None
    if candidate_mode == "stable_shapes":
        stable_fourier_window_contract = _validate_stable_fourier_profiles(
            captured["estep_meta"],
            enabled=candidate_enabled,
            label=label,
            image_shape=tuple(int(value) for value in dataset.image_shape),
        )
    if candidate_mode == "all_optimized_stable_shapes":
        stable_pair_contract = _validate_all_optimized_stable_pair_profiles(
            captured["estep_meta"],
            enabled=candidate_enabled,
            label=label,
            image_shape=tuple(int(value) for value in dataset.image_shape),
        )
        stable_fourier_window_contract = stable_pair_contract[
            "stable_fourier"
        ]
        stable_flat_capacity_contract = stable_pair_contract[
            "stable_flat_capacity"
        ]
    if candidate_mode == "stable_flat_capacity":
        for key in (
            "requested_stable_flat_row_capacity",
            "effective_stable_flat_row_capacity",
        ):
            if key not in captured["estep_meta"]:
                raise RuntimeError(f"{label} did not report {key}")
            if bool(captured["estep_meta"][key]) != bool(candidate_enabled):
                raise RuntimeError(
                    f"{label} reported {key}="
                    f"{captured['estep_meta'][key]!r}, expected {candidate_enabled!r}"
                )
        stable_flat_capacity_contract = _validate_stable_flat_capacity_profiles(
            captured["estep_meta"],
            enabled=candidate_enabled,
            label=label,
        )
    execution_contract = _validate_arm_execution_contract(
        candidate_mode=candidate_mode,
        candidate_enabled=candidate_enabled,
        requested_environment=requested_environment,
        effective_environment=effective_environment,
        estep_meta=captured["estep_meta"],
        backend_mode=resolved_backend_mode,
    )
    if candidate_mode == "all_optimized_stable_shapes":
        execution_contract["all_optimized_stable_pair"] = stable_pair_contract
        execution_contract["all_optimized_stable_pair_profile_exact"] = True
    if candidate_mode in {
        "all_optimized",
        "exact_coarse_single_translate",
        "exact_compact_preprocess",
        "fused_pair_fine_score",
    }:
        all_optimized_enabled = bool(
            candidate_enabled
            if candidate_mode == "all_optimized"
            else True
        )
        all_optimized_contract = _validate_all_optimized_profiles(
            captured["estep_meta"],
            enabled=all_optimized_enabled,
            label=label,
            image_shape=tuple(int(value) for value in dataset.image_shape),
        )
        execution_contract["all_optimized"] = all_optimized_contract
        execution_contract["all_optimized_profile_exact"] = True
        stable_fourier_window_contract = all_optimized_contract["stable_fourier"]
        if all_optimized_enabled:
            stable_flat_capacity_contract = all_optimized_contract[
                "stable_flat_capacity"
            ]
    if candidate_mode == "exact_coarse_single_translate":
        if exact_coarse_profile_enabled:
            execution_contract["exact_coarse_single_translate"] = (
                _validate_exact_coarse_single_translate_profiles(
                    captured["estep_meta"],
                    enabled=candidate_enabled,
                    label=label,
                )
            )
        else:
            if any(
                isinstance(value, dict)
                and "exact_coarse_operand_assembly" in value
                for value in captured["estep_meta"].values()
            ):
                raise RuntimeError(
                    f"{label} profile-off timing arm published call-count metadata",
                )
            execution_contract["exact_coarse_single_translate"] = {
                "enabled": bool(candidate_enabled),
                "profile_enabled": False,
                "profile_checked": False,
                "profile_free_wall_timing": True,
            }
    if candidate_mode == "exact_compact_preprocess":
        if exact_coarse_profile_enabled:
            execution_contract["exact_compact_preprocess"] = (
                _validate_exact_compact_preprocess_profiles(
                    captured["estep_meta"],
                    enabled=candidate_enabled,
                    label=label,
                )
            )
        else:
            if any(
                isinstance(value, dict)
                and "exact_coarse_operand_assembly" in value
                for value in captured["estep_meta"].values()
            ):
                raise RuntimeError(
                    f"{label} profile-off timing arm published call-count metadata",
                )
            execution_contract["exact_compact_preprocess"] = {
                "enabled": bool(candidate_enabled),
                "profile_enabled": False,
                "profile_checked": False,
                "profile_free_wall_timing": True,
            }
    if candidate_mode == "fused_pair_fine_score":
        execution_contract["fused_pair_fine_score"] = (
            _validate_fused_pair_fine_profiles(
                captured["estep_meta"],
                enabled=candidate_enabled,
                label=label,
            )
        )
    if hybrid_image_batch_request is not None:
        execution_contract["hybrid_image_batch"] = (
            _validate_hybrid_image_batch_profiles(
                captured["estep_meta"],
                requested_batch_size=hybrid_image_batch_request,
                label=label,
            )
        )
    performance_summary = _arm_performance_summary(
        wall_s=wall_s,
        estep_meta=captured["estep_meta"],
        persistent_cache=persistent_cache,
    )
    return {
        "label": label,
        "candidate_mode": candidate_mode,
        "candidate_enabled": bool(candidate_enabled),
        "hybrid": bool(
            _candidate_uses_hybrid(
                candidate_mode if backend_mode is None else resolved_backend_mode
            )
            and (
                candidate_enabled
                or (
                    backend_mode is None
                    and candidate_mode
                    in {
                        "stable_shapes",
                        "stable_flat_capacity",
                        "exact_coarse_single_translate",
                        "exact_compact_preprocess",
                        "fused_pair_fine_score",
                    }
                )
            )
        ),
        "backend_mode": resolved_backend_mode,
        "execution_contract": execution_contract,
        "performance_summary": performance_summary,
        "wall_s": wall_s,
        "wall_clock_started_epoch_s": wall_clock_started_epoch_s,
        "wall_clock_ended_epoch_s": wall_clock_ended_epoch_s,
        "persistent_cache": persistent_cache,
        "stable_fourier_window_contract": stable_fourier_window_contract,
        "stable_flat_capacity_contract": stable_flat_capacity_contract,
        "initial_state_manifest": initial_state_manifest,
        "initial_particle_state_manifest": initial_particle_state_manifest,
        "initial_sampling_state_manifest": initial_sampling_state_manifest,
        "final_state": final_state,
        "particle_state": particle_state,
        "sampling_state": sampling_state,
        **captured,
    }


def _pair_report(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "left": left["label"],
        "right": right["label"],
        "estep_meta": _meta_comparison(left["estep_meta"], right["estep_meta"]),
        "support_audits": _support_audit_comparison(left["estep_meta"], right["estep_meta"]),
        "accumulators": _accumulator_comparison(left["accumulators"], right["accumulators"]),
        "particle_state": _dataclass_comparison(left["particle_state"], right["particle_state"]),
        "sampling_state": _dataclass_comparison(left["sampling_state"], right["sampling_state"]),
        "final_state": _dataclass_comparison(left["final_state"], right["final_state"]),
    }


def _comparison_fields_exact(section: dict[str, Any]) -> bool:
    return bool(section) and all(
        isinstance(value, dict) and value.get("exact_equal") is True
        for value in section.values()
    )


def _numeric_comparison_values(
    section: dict[str, Any],
    *,
    nested_entries: bool,
) -> dict[str, float]:
    entries = section.get("entries", []) if nested_entries else [section]
    values: dict[str, float] = {}
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        for field, comparison in entry.items():
            if not isinstance(comparison, dict):
                continue
            value = comparison.get("normalized_l2_delta")
            if value is None:
                continue
            numeric = float(value)
            if not np.isfinite(numeric) or numeric < 0.0:
                raise RuntimeError("same-state comparison contains an invalid delta")
            prefix = f"{entry_index}." if nested_entries else ""
            values[f"{prefix}{field}"] = numeric
    return values


def _repeat_envelope_report(
    comparisons: dict[str, dict[str, Any]],
    *,
    repeat_pairs: tuple[str, str],
    cross_pairs: tuple[str, ...],
    section: str,
    nested_entries: bool,
) -> dict[str, Any]:
    repeat_values = {
        pair: _numeric_comparison_values(
            comparisons[pair][section],
            nested_entries=nested_entries,
        )
        for pair in repeat_pairs
    }
    cross_values = {
        pair: _numeric_comparison_values(
            comparisons[pair][section],
            nested_entries=nested_entries,
        )
        for pair in cross_pairs
    }
    paths = sorted(
        set().union(
            *(set(values) for values in (*repeat_values.values(), *cross_values.values()))
        )
    )
    rows: dict[str, Any] = {}
    for path in paths:
        direct_repeat = repeat_values[repeat_pairs[0]].get(path, 0.0)
        candidate_repeat = repeat_values[repeat_pairs[1]].get(path, 0.0)
        envelope = max(direct_repeat, candidate_repeat)
        cross_by_pair = {
            pair: values.get(path, 0.0) for pair, values in cross_values.items()
        }
        cross_max = max(cross_by_pair.values(), default=0.0)
        within = (
            cross_max <= float(np.nextafter(envelope, np.inf))
            if envelope > 0.0
            else cross_max == 0.0
        )
        rows[path] = {
            "direct_repeat_normalized_l2": direct_repeat,
            "candidate_repeat_normalized_l2": candidate_repeat,
            "repeat_envelope_normalized_l2": envelope,
            "cross_normalized_l2_by_pair": cross_by_pair,
            "maximum_cross_normalized_l2": cross_max,
            "maximum_cross_over_repeat_envelope": (
                cross_max / envelope if envelope > 0.0 else None
            ),
            "within_observed_repeat_envelope": within,
        }
    outside = [
        path
        for path, row in rows.items()
        if not row["within_observed_repeat_envelope"]
    ]
    return {
        "policy": (
            "compare each normalized-L2 field against the maximum of the "
            "direct/direct and candidate/candidate observed repeats"
        ),
        "all_cross_within_observed_repeat_envelope": not outside,
        "outside_paths": outside,
        "rows": rows,
    }


def _compact_science_contract(
    comparisons: dict[str, dict[str, Any]],
    arm_order: tuple[str, str, str, str],
) -> dict[str, Any]:
    """Summarize hard discrete parity and observed continuous repeat noise."""

    direct_repeat = f"{arm_order[0]}__vs__{arm_order[3]}"
    candidate_repeat = f"{arm_order[1]}__vs__{arm_order[2]}"
    cross_pairs = (
        f"{arm_order[0]}__vs__{arm_order[1]}",
        f"{arm_order[0]}__vs__{arm_order[2]}",
        f"{arm_order[3]}__vs__{arm_order[1]}",
        f"{arm_order[3]}__vs__{arm_order[2]}",
    )
    required_meta = (
        "selected_particle_ids",
        "best_pose_rotation_ids",
        "pose_assignments",
        "class_assignments",
        "best_pose_translations",
        "significant_counts",
    )
    exact_checks: dict[str, Any] = {}
    for pair in cross_pairs:
        comparison = comparisons[pair]
        meta = comparison["estep_meta"]
        missing_meta = [key for key in required_meta if key not in meta]
        unequal_meta = [
            key
            for key in required_meta
            if key in meta and meta[key].get("exact_equal") is not True
        ]
        exact_checks[pair] = {
            "required_meta_present": not missing_meta,
            "missing_meta": missing_meta,
            "required_meta_exact": not unequal_meta,
            "unequal_meta": unequal_meta,
            "support_audits_exact": comparison["support_audits"].get(
                "exact_equal",
            )
            is True,
            "particle_state_exact": _comparison_fields_exact(
                comparison["particle_state"],
            ),
            "sampling_state_exact": _comparison_fields_exact(
                comparison["sampling_state"],
            ),
        }
        exact_checks[pair]["pass"] = all(
            value is True
            for key, value in exact_checks[pair].items()
            if key
            not in {
                "missing_meta",
                "unequal_meta",
            }
        )
    exact_pass = all(row["pass"] for row in exact_checks.values())
    repeat_pairs = (direct_repeat, candidate_repeat)
    result = {
        "hard_exact_contract_passed": exact_pass,
        "hard_exact_policy": (
            "all crossed pairs require exact selected IDs, pose/translation/class "
            "decisions, significant counts, complete support audits, particle "
            "state, and sampling state"
        ),
        "exact_cross_pair_checks": exact_checks,
        "accumulator_repeat_envelope": _repeat_envelope_report(
            comparisons,
            repeat_pairs=repeat_pairs,
            cross_pairs=cross_pairs,
            section="accumulators",
            nested_entries=True,
        ),
        "final_state_repeat_envelope": _repeat_envelope_report(
            comparisons,
            repeat_pairs=repeat_pairs,
            cross_pairs=cross_pairs,
            section="final_state",
            nested_entries=False,
        ),
    }
    result["scalar_meta_repeat_envelope"] = _repeat_envelope_report(
        comparisons,
        repeat_pairs=repeat_pairs,
        cross_pairs=cross_pairs,
        section="estep_meta",
        nested_entries=False,
    )
    result["atomic_repeat_envelope_passed"] = bool(
        result["accumulator_repeat_envelope"][
            "all_cross_within_observed_repeat_envelope"
        ]
        and result["scalar_meta_repeat_envelope"][
            "all_cross_within_observed_repeat_envelope"
        ]
        and result["final_state_repeat_envelope"][
            "all_cross_within_observed_repeat_envelope"
        ]
    )
    return result


def _exact_coarse_single_translate_runtime_contract(
    arms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare warmed ABBA timings with all other optimized seams fixed."""

    result: dict[str, Any] = {}
    environments: dict[str, dict[str, str]] = {}
    for backend, token in (("skip_off", "_off_"), ("skip_on", "_on_")):
        labels = tuple(
            label
            for label in EXACT_COARSE_SINGLE_TRANSLATE_ARM_ORDER
            if token in label
        )
        measurements: dict[str, list[float]] = {}
        for metric in ("wall_s", "pass1_time_s", "pass2_time_s"):
            values = []
            for label in labels:
                source = (
                    arms[label]
                    if metric == "wall_s"
                    else arms[label]["performance_summary"]
                )
                if metric not in source:
                    raise RuntimeError(f"{label} omitted runtime metric {metric}")
                values.append(float(source[metric]))
            measurements[metric] = values
        result[backend] = {
            "labels": list(labels),
            "repeat_count": len(labels),
            "measurements": measurements,
            "median": {
                metric: float(np.median(values))
                for metric, values in measurements.items()
            },
        }
        for label in labels:
            requested = dict(
                arms[label]["execution_contract"]["requested_environment"],
            )
            requested.pop(EXACT_COARSE_SINGLE_TRANSLATE_ENVIRONMENT)
            environments[label] = requested

    distinct_environments = {
        json.dumps(environment, sort_keys=True, separators=(",", ":"))
        for environment in environments.values()
    }
    if len(distinct_environments) != 1:
        raise RuntimeError(
            "single-translate arms differ outside the isolated environment flag",
        )
    changes = {}
    for metric in ("wall_s", "pass1_time_s", "pass2_time_s"):
        control = result["skip_off"]["median"][metric]
        candidate = result["skip_on"]["median"][metric]
        changes[metric] = {
            "skip_off_median": control,
            "skip_on_median": candidate,
            "fractional_change": candidate / control - 1.0,
            "speedup": control / candidate,
        }
    result["skip_on_vs_skip_off"] = changes
    result["isolated_environment_contract"] = {
        "only_difference": EXACT_COARSE_SINGLE_TRANSLATE_ENVIRONMENT,
        "all_other_environment_exact": True,
        "profile_free_wall_timing": True,
        "profile_counter_device_synchronization": False,
    }
    return result


def _exact_compact_preprocess_runtime_contract(
    arms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare warmed exact preprocessing ABBA timings at one frozen state."""

    result: dict[str, Any] = {}
    environments: dict[str, dict[str, str]] = {}
    for backend, token in (
        ("preprocess_off", "_off_"),
        ("preprocess_on", "_on_"),
    ):
        labels = tuple(
            label
            for label in EXACT_COMPACT_PREPROCESS_ARM_ORDER
            if token in label
        )
        measurements: dict[str, list[float]] = {}
        for metric in ("wall_s", "pass1_time_s", "pass2_time_s"):
            values = []
            for label in labels:
                source = (
                    arms[label]
                    if metric == "wall_s"
                    else arms[label]["performance_summary"]
                )
                if metric not in source:
                    raise RuntimeError(f"{label} omitted runtime metric {metric}")
                values.append(float(source[metric]))
            measurements[metric] = values
        result[backend] = {
            "labels": list(labels),
            "repeat_count": len(labels),
            "measurements": measurements,
            "median": {
                metric: float(np.median(values))
                for metric, values in measurements.items()
            },
        }
        for label in labels:
            requested = dict(
                arms[label]["execution_contract"]["requested_environment"],
            )
            requested.pop(EXACT_COMPACT_PREPROCESS_ENVIRONMENT)
            environments[label] = requested

    distinct_environments = {
        json.dumps(environment, sort_keys=True, separators=(",", ":"))
        for environment in environments.values()
    }
    if len(distinct_environments) != 1:
        raise RuntimeError(
            "exact compact preprocessing arms differ outside the isolated flag",
        )
    changes = {}
    for metric in ("wall_s", "pass1_time_s", "pass2_time_s"):
        control = result["preprocess_off"]["median"][metric]
        candidate = result["preprocess_on"]["median"][metric]
        changes[metric] = {
            "preprocess_off_median": control,
            "preprocess_on_median": candidate,
            "fractional_change": candidate / control - 1.0,
            "speedup": control / candidate,
        }
    result["preprocess_on_vs_preprocess_off"] = changes
    result["isolated_environment_contract"] = {
        "only_difference": EXACT_COMPACT_PREPROCESS_ENVIRONMENT,
        "all_other_environment_exact": True,
        "profile_free_wall_timing": True,
        "profile_counter_device_synchronization": False,
    }
    return result


HYBRID_IMAGE_BATCH_REQUIRED_META = (
    "selected_particle_ids",
    "best_pose_rotation_ids",
    "pose_assignments",
    "class_assignments",
    "best_pose_translations",
    "max_posterior_per_image",
    "significant_counts",
)
HYBRID_IMAGE_BATCH_PUBLIC_STATE = (
    "Mavg",
    "ave_Pmax",
    "pdf_class",
    "pdf_direction",
    "sigma2_offset",
    "tau2_class",
    "fsc_halves_class",
    "current_resolution",
    "current_size",
)


def _pair_name(
    comparisons: dict[str, dict[str, Any]],
    left: str,
    right: str,
) -> str:
    forward = f"{left}__vs__{right}"
    reverse = f"{right}__vs__{left}"
    if forward in comparisons:
        return forward
    if reverse in comparisons:
        return reverse
    raise KeyError(f"missing same-state comparison for {left!r} and {right!r}")


def _unordered_pairs(labels: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (left, right)
        for left_index, left in enumerate(labels)
        for right in labels[left_index + 1 :]
    )


def _accumulator_scalar_fields_exact(section: dict[str, Any]) -> bool:
    entries = section.get("entries")
    if section.get("comparable") is not True or not isinstance(entries, list) or not entries:
        return False
    scalar_count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            return False
        for comparison in entry.values():
            if not isinstance(comparison, dict) or "left_shape" in comparison:
                continue
            scalar_count += 1
            if comparison.get("exact_equal") is not True:
                return False
    return scalar_count > 0


def _dataclass_scalar_fields_exact(section: dict[str, Any]) -> bool:
    scalar_fields = [
        comparison
        for comparison in section.values()
        if isinstance(comparison, dict) and "left_shape" not in comparison
    ]
    return bool(scalar_fields) and all(
        comparison.get("exact_equal") is True for comparison in scalar_fields
    )


def _multi_repeat_envelope_report(
    comparisons: dict[str, dict[str, Any]],
    *,
    control_labels: tuple[str, ...],
    candidate_labels: tuple[str, ...],
    section: str,
    nested_entries: bool,
) -> dict[str, Any]:
    control_pairs = _unordered_pairs(control_labels)
    candidate_pairs = _unordered_pairs(candidate_labels)
    cross_pairs = tuple(
        (control, candidate)
        for control in control_labels
        for candidate in candidate_labels
    )

    def values_by_pair(pairs):
        return {
            _pair_name(comparisons, left, right): _numeric_comparison_values(
                comparisons[_pair_name(comparisons, left, right)][section],
                nested_entries=nested_entries,
            )
            for left, right in pairs
        }

    control_values = values_by_pair(control_pairs)
    candidate_values = values_by_pair(candidate_pairs)
    cross_values = values_by_pair(cross_pairs)
    paths = sorted(
        set().union(
            *(
                set(values)
                for values in (
                    *control_values.values(),
                    *candidate_values.values(),
                    *cross_values.values(),
                )
            ),
        ),
    )
    rows: dict[str, Any] = {}
    for path in paths:
        control_max = max(
            (values.get(path, 0.0) for values in control_values.values()),
            default=0.0,
        )
        candidate_max = max(
            (values.get(path, 0.0) for values in candidate_values.values()),
            default=0.0,
        )
        envelope = max(control_max, candidate_max)
        cross_by_pair = {
            pair: values.get(path, 0.0) for pair, values in cross_values.items()
        }
        cross_max = max(cross_by_pair.values(), default=0.0)
        within = (
            cross_max <= float(np.nextafter(envelope, np.inf))
            if envelope > 0.0
            else cross_max == 0.0
        )
        rows[path] = {
            "control_repeat_max_normalized_l2": control_max,
            "candidate_repeat_max_normalized_l2": candidate_max,
            "repeat_envelope_normalized_l2": envelope,
            "maximum_cross_normalized_l2": cross_max,
            "cross_normalized_l2_by_pair": cross_by_pair,
            "within_observed_repeat_envelope": within,
        }
    outside = [
        path for path, row in rows.items() if not row["within_observed_repeat_envelope"]
    ]
    return {
        "policy": (
            "every 110/200 cross delta must be no larger than the maximum "
            "within-backend delta over all six pairs among four warm repeats"
        ),
        "control_repeat_pair_count": len(control_pairs),
        "candidate_repeat_pair_count": len(candidate_pairs),
        "cross_pair_count": len(cross_pairs),
        "all_cross_within_observed_repeat_envelope": not outside,
        "outside_paths": outside,
        "rows": rows,
    }


def _hybrid_image_batch_science_contract(
    comparisons: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Gate exact science and repeat-bounded atomics for B=110 versus B=200."""

    control_labels = tuple(
        label for label in HYBRID_IMAGE_BATCH_ARM_ORDER if "batch110" in label
    )
    candidate_labels = tuple(
        label for label in HYBRID_IMAGE_BATCH_ARM_ORDER if "batch200" in label
    )
    exact_pairs = _unordered_pairs(HYBRID_IMAGE_BATCH_GATE_ARM_ORDER)
    exact_checks: dict[str, Any] = {}
    for left, right in exact_pairs:
        pair = _pair_name(comparisons, left, right)
        comparison = comparisons[pair]
        meta = comparison["estep_meta"]
        missing_meta = [
            key for key in HYBRID_IMAGE_BATCH_REQUIRED_META if key not in meta
        ]
        unequal_meta = [
            key
            for key in HYBRID_IMAGE_BATCH_REQUIRED_META
            if key in meta and meta[key].get("exact_equal") is not True
        ]
        final_state = comparison["final_state"]
        missing_public_state = [
            key for key in HYBRID_IMAGE_BATCH_PUBLIC_STATE if key not in final_state
        ]
        unequal_public_state = [
            key
            for key in HYBRID_IMAGE_BATCH_PUBLIC_STATE
            if key in final_state
            and final_state[key].get("exact_equal") is not True
        ]
        exact_checks[pair] = {
            "required_meta_present": not missing_meta,
            "missing_meta": missing_meta,
            "required_meta_exact": not unequal_meta,
            "unequal_meta": unequal_meta,
            "support_audits_exact": comparison["support_audits"].get(
                "exact_equal",
            )
            is True,
            "accumulator_scalars_exact": _accumulator_scalar_fields_exact(
                comparison["accumulators"],
            ),
            "particle_state_exact": _comparison_fields_exact(
                comparison["particle_state"],
            ),
            "sampling_state_exact": _comparison_fields_exact(
                comparison["sampling_state"],
            ),
            "final_state_scalars_exact": _dataclass_scalar_fields_exact(final_state),
            "public_state_present": not missing_public_state,
            "missing_public_state": missing_public_state,
            "public_state_exact": not unequal_public_state,
            "unequal_public_state": unequal_public_state,
        }
        exact_checks[pair]["pass"] = all(
            value is True
            for key, value in exact_checks[pair].items()
            if key
            not in {
                "missing_meta",
                "unequal_meta",
                "missing_public_state",
                "unequal_public_state",
            }
        )

    accumulator_envelope = _multi_repeat_envelope_report(
        comparisons,
        control_labels=control_labels,
        candidate_labels=candidate_labels,
        section="accumulators",
        nested_entries=True,
    )
    final_state_envelope = _multi_repeat_envelope_report(
        comparisons,
        control_labels=control_labels,
        candidate_labels=candidate_labels,
        section="final_state",
        nested_entries=False,
    )
    return {
        "hard_exact_contract_passed": all(
            row["pass"] for row in exact_checks.values()
        ),
        "hard_exact_policy": (
            "all 36 oracle/repeat/cross pairs require exact support, hard and "
            "posterior decisions, accumulator/state scalars, particle/sampling "
            "state, and public state"
        ),
        "exact_pair_checks": exact_checks,
        "accumulator_repeat_envelope": accumulator_envelope,
        "final_state_repeat_envelope": final_state_envelope,
        "atomic_repeat_envelope_passed": bool(
            accumulator_envelope["all_cross_within_observed_repeat_envelope"]
            and final_state_envelope["all_cross_within_observed_repeat_envelope"]
        ),
    }


def _hybrid_image_batch_runtime_contract(
    arms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare all four warmed observations from each mirrored backend."""

    result: dict[str, Any] = {}
    for backend, token in (("batch110", "batch110"), ("batch200", "batch200")):
        labels = tuple(label for label in HYBRID_IMAGE_BATCH_ARM_ORDER if token in label)
        measurements: dict[str, list[float]] = {}
        for metric in ("wall_s", "pass1_time_s", "pass2_time_s"):
            values = []
            for label in labels:
                source = (
                    arms[label]
                    if metric == "wall_s"
                    else arms[label]["performance_summary"]
                )
                if metric not in source:
                    raise RuntimeError(f"{label} omitted runtime metric {metric}")
                values.append(float(source[metric]))
            measurements[metric] = values
        result[backend] = {
            "labels": list(labels),
            "repeat_count": len(labels),
            "measurements": measurements,
            "median": {
                metric: float(np.median(values))
                for metric, values in measurements.items()
            },
        }
    changes = {}
    for metric in ("wall_s", "pass1_time_s", "pass2_time_s"):
        control = result["batch110"]["median"][metric]
        candidate = result["batch200"]["median"][metric]
        changes[metric] = {
            "batch110_median": control,
            "batch200_median": candidate,
            "fractional_change": candidate / control - 1.0,
            "speedup": control / candidate,
        }
    result["batch200_vs_batch110"] = changes
    return result


def _fused_pair_fine_runtime_contract(
    arms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare warmed selected-pair scoring with all other seams frozen on."""

    metrics = (
        "wall_s",
        "pass1_time_s",
        "pass2_time_s",
        "local_em_time_s",
        "local_big_jit_bucket_s",
        "local_pack_s",
        "local_noise_s",
        "local_postprocess_s",
        "local_final_accumulator_s",
        "local_unattributed_em_time_s",
    )
    result: dict[str, Any] = {}
    environments: dict[str, dict[str, str]] = {}
    timed_cache_stable = True
    for backend, token in (("pair_off", "_off_"), ("pair_on", "_on_")):
        labels = tuple(
            label for label in FUSED_PAIR_FINE_SCORE_ARM_ORDER if token in label
        )
        measurements: dict[str, list[float]] = {}
        for metric in metrics:
            values = []
            for label in labels:
                source = arms[label] if metric == "wall_s" else arms[label][
                    "performance_summary"
                ]
                if metric not in source:
                    raise RuntimeError(f"{label} omitted runtime metric {metric}")
                values.append(float(source[metric]))
            measurements[metric] = values
        cache = {}
        pair_profiles = {}
        for label in labels:
            cache[label] = arms[label]["persistent_cache"]
            timed_cache_stable = bool(
                timed_cache_stable
                and cache[label]["delta"]["no_new_target_family_programs"]
            )
            pair_profiles[label] = arms[label]["execution_contract"][
                "fused_pair_fine_score"
            ]
            requested = dict(
                arms[label]["execution_contract"]["requested_environment"],
            )
            requested.pop(FUSED_PAIR_FINE_SCORE_ENVIRONMENT)
            environments[label] = requested
        result[backend] = {
            "labels": list(labels),
            "repeat_count": len(labels),
            "measurements": measurements,
            "median": {
                metric: float(np.median(values))
                for metric, values in measurements.items()
            },
            "persistent_cache": cache,
            "pair_profiles": pair_profiles,
        }

    distinct_environments = {
        json.dumps(environment, sort_keys=True, separators=(",", ":"))
        for environment in environments.values()
    }
    if len(distinct_environments) != 1:
        raise RuntimeError(
            "fused-pair arms differ outside the isolated environment flag",
        )
    changes = {}
    for metric in metrics:
        control = result["pair_off"]["median"][metric]
        candidate = result["pair_on"]["median"][metric]
        changes[metric] = {
            "pair_off_median": control,
            "pair_on_median": candidate,
            "fractional_change": candidate / control - 1.0 if control else None,
            "speedup": control / candidate if candidate else None,
        }
    final_snapshot = arms[FUSED_PAIR_FINE_SCORE_ARM_ORDER[-1]][
        "persistent_cache"
    ]["after"]
    result["pair_on_vs_pair_off"] = changes
    result["material_speedup_threshold"] = 1.05
    result["material_speedup_passed"] = bool(
        changes["wall_s"]["speedup"] is not None
        and changes["wall_s"]["speedup"] >= result["material_speedup_threshold"]
    )
    result["persistent_cache_contract"] = {
        "target_families": list(PERSISTENT_CACHE_TARGET_FAMILIES),
        "timed_arms_add_no_target_family_programs_after_prewarm": (
            timed_cache_stable
        ),
        "final_target_family_counts": final_snapshot["target_family_counts"],
        "final_target_family_bytes": final_snapshot["target_family_bytes"],
    }
    result["isolated_environment_contract"] = {
        "only_difference": FUSED_PAIR_FINE_SCORE_ENVIRONMENT,
        "all_other_environment_exact": True,
        "profile_free_wall_timing": True,
        "profile_counter_device_synchronization": False,
    }
    return result


def _gpu_memory_report(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Attribute the line-buffered one-second nvidia-smi samples to each arm."""

    raw_path = os.environ.get(GPU_MONITOR_ENVIRONMENT, "").strip()
    if not raw_path:
        return {"available": False, "path": None, "arms": {}}
    path = Path(raw_path)
    if not path.is_file():
        return {"available": False, "path": str(path), "arms": {}}
    samples: list[tuple[float, int, int, int]] = []
    with path.open(newline="") as handle:
        for raw_row in csv.DictReader(handle):
            row = {str(key).strip(): str(value).strip() for key, value in raw_row.items()}
            try:
                timestamp = dt.datetime.strptime(
                    row["timestamp"],
                    "%Y/%m/%d %H:%M:%S.%f",
                ).timestamp()
                used_mib = int(row["memory.used [MiB]"].split()[0])
                total_mib = int(row["memory.total [MiB]"].split()[0])
                utilization = int(row["utilization.gpu [%]"].split()[0])
            except (KeyError, ValueError):
                continue
            samples.append((timestamp, used_mib, total_mib, utilization))
    arm_rows: dict[str, Any] = {}
    for label, arm in arms.items():
        start = float(arm["wall_clock_started_epoch_s"])
        end = float(arm["wall_clock_ended_epoch_s"])
        selected = [sample for sample in samples if start <= sample[0] <= end]
        arm_rows[label] = {
            "sample_count": len(selected),
            "peak_memory_used_mib": (
                max(sample[1] for sample in selected) if selected else None
            ),
            "minimum_memory_used_mib": (
                min(sample[1] for sample in selected) if selected else None
            ),
            "memory_total_mib": selected[0][2] if selected else None,
            "peak_utilization_percent": (
                max(sample[3] for sample in selected) if selected else None
            ),
            "window_started_epoch_s": start,
            "window_ended_epoch_s": end,
        }
    return {
        "available": bool(samples),
        "path": str(path.resolve()),
        "sampling_period_s": 1,
        "allocator_caveat": (
            "JAX retains allocations, so later-arm HBM is a process high-water "
            "comparison rather than isolated allocation accounting"
        ),
        "all_arms_sampled": bool(arm_rows)
        and all(row["sample_count"] > 0 for row in arm_rows.values()),
        "total_sample_count": len(samples),
        "arms": arm_rows,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if HYBRID_IMAGE_BATCH_ENVIRONMENT in os.environ:
        raise ValueError(
            f"{HYBRID_IMAGE_BATCH_ENVIRONMENT} must be absent from the outer "
            "environment; the mirrored harness scopes it per arm",
        )
    if EXACT_COARSE_SINGLE_TRANSLATE_ENVIRONMENT in os.environ:
        raise ValueError(
            f"{EXACT_COARSE_SINGLE_TRANSLATE_ENVIRONMENT} must be absent from "
            "the outer environment; the same-state harness scopes it per arm",
        )
    if EXACT_COARSE_ASSEMBLY_PROFILE_ENVIRONMENT in os.environ:
        raise ValueError(
            f"{EXACT_COARSE_ASSEMBLY_PROFILE_ENVIRONMENT} must be absent from "
            "the outer environment; the same-state harness scopes it per arm",
        )
    if EXACT_COMPACT_PREPROCESS_ENVIRONMENT in os.environ:
        raise ValueError(
            f"{EXACT_COMPACT_PREPROCESS_ENVIRONMENT} must be absent from the "
            "outer environment; the same-state harness scopes it per arm",
        )
    if FUSED_PAIR_FINE_SCORE_ENVIRONMENT in os.environ:
        raise ValueError(
            f"{FUSED_PAIR_FINE_SCORE_ENVIRONMENT} must be absent from the "
            "outer environment; the same-state harness scopes it per arm",
        )
    fixture_dir = args.fixture_dir.resolve(strict=True)
    acceptance_path = args.acceptance_config.resolve(strict=True)
    output_root = args.output_root.resolve()
    native_checkpoint = _resolve_native_checkpoint_inputs(
        optimiser=args.native_checkpoint_optimiser,
        data_star=args.native_checkpoint_data_star,
        data_dir=args.native_data_dir,
    )
    target_image_name = (
        None if args.target_image_name is None else str(args.target_image_name).strip()
    )
    if args.target_image_name is not None and not target_image_name:
        raise ValueError("target-image-name must be non-empty")
    if args.checkpoint_iteration < 1:
        raise ValueError("checkpoint-iteration must be positive")
    if args.image_batch_size < 1:
        raise ValueError("image-batch-size must be positive")
    if args.mirrored_incremental_panels and args.mirrored_hybrid_image_batch_panels:
        raise ValueError("mirrored panel modes are mutually exclusive")
    if args.mirrored_incremental_panels and args.candidate_mode != "packed_final_noise":
        raise ValueError(
            "mirrored incremental panels require --candidate-mode packed_final_noise"
        )
    if (
        args.mirrored_hybrid_image_batch_panels
        and args.candidate_mode != "all_optimized"
    ):
        raise ValueError(
            "mirrored hybrid image-batch panels require --candidate-mode all_optimized",
        )
    if (
        args.candidate_mode == "fused_coarse_projector"
        and args.coarse_prefix_dump_original_index is not None
    ):
        raise ValueError(
            "fused-coarse projector mode does not support compact-hybrid coarse-prefix dumps"
        )
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    acceptance = json.loads(acceptance_path.read_text())
    frozen_nr_iter = int(acceptance["science_contract"]["definition"]["nr_iter"])
    if args.checkpoint_iteration >= frozen_nr_iter:
        raise ValueError("checkpoint-iteration must leave one next iteration")

    checkpoint_started = time.perf_counter()
    checkpoint = _capture_direct_checkpoint(
        fixture_dir=fixture_dir,
        acceptance=acceptance,
        output_root=output_root,
        checkpoint_iteration=args.checkpoint_iteration,
        image_batch_size=args.image_batch_size,
        checkpoint_candidate_mode=(
            "all_optimized_stable_shapes"
            if native_checkpoint is None
            and args.candidate_mode == "all_optimized_stable_shapes"
            else "hybrid"
        ),
        checkpoint_candidate_enabled=(
            native_checkpoint is None
            and args.candidate_mode == "all_optimized_stable_shapes"
        ),
        native_checkpoint_optimiser=(
            None if native_checkpoint is None else native_checkpoint["optimiser"]
        ),
        native_checkpoint_data_star=(
            None if native_checkpoint is None else native_checkpoint["data_star"]
        ),
        native_data_dir=(
            None if native_checkpoint is None else native_checkpoint["data_dir"]
        ),
    )
    checkpoint_wall_s = float(time.perf_counter() - checkpoint_started)
    target_checkpoint: dict[str, Any] | None = None
    target_row_index: int | None = None
    if target_image_name is not None:
        from recovar.data_io.starfile import read_star

        checkpoint_star, _checkpoint_optics = read_star(checkpoint["opts"].fn_img)
        target_row_index = _resolve_target_particle_index(
            checkpoint_star,
            target_image_name,
        )
        target_checkpoint = {
            "image_name": target_image_name,
            "checkpoint_row_index": target_row_index,
            "source_star": _target_star_metadata(
                checkpoint_star,
                target_row_index,
            ),
            "particle_state": _target_particle_summary(
                checkpoint["particle_state"],
                None,
                row_index=target_row_index,
                pixel_size=float(checkpoint["dataset"].voxel_size),
            ),
        }
    checkpoint_manifest = {
        "state": _dataclass_manifest(checkpoint["result"].state),
        "particle_state": _dataclass_manifest(checkpoint["particle_state"]),
        "sampling_state": _dataclass_manifest(checkpoint["sampling_state"]),
    }
    expected_manifests = {
        "initial_state_manifest": checkpoint_manifest["state"]["manifest_sha256"],
        "initial_particle_state_manifest": checkpoint_manifest["particle_state"]["manifest_sha256"],
        "initial_sampling_state_manifest": checkpoint_manifest["sampling_state"]["manifest_sha256"],
    }

    prewarm: dict[str, Any] = {}
    if args.mirrored_incremental_panels:
        for backend_mode in ("packed_deferred", "packed_final_noise"):
            warm = _run_transition_arm(
                checkpoint,
                label=f"prewarm_{backend_mode}",
                candidate_mode=args.candidate_mode,
                candidate_enabled=backend_mode == args.candidate_mode,
                checkpoint_iteration=args.checkpoint_iteration,
                backend_mode=backend_mode,
            )
            for key, expected in expected_manifests.items():
                if warm[key]["manifest_sha256"] != expected:
                    raise RuntimeError(
                        f"prewarm_{backend_mode} did not start from the exact shared {key}"
                    )
            prewarm[backend_mode] = {
                "label": warm["label"],
                "backend_mode": warm["backend_mode"],
                "wall_s": warm["wall_s"],
                "initial_state_manifest_sha256": warm["initial_state_manifest"][
                    "manifest_sha256"
                ],
                "initial_particle_state_manifest_sha256": warm[
                    "initial_particle_state_manifest"
                ]["manifest_sha256"],
                "initial_sampling_state_manifest_sha256": warm[
                    "initial_sampling_state_manifest"
                ]["manifest_sha256"],
            }
            del warm
            gc.collect()
        arm_specs = _incremental_arm_specs()
    elif args.mirrored_hybrid_image_batch_panels:
        for batch_label, batch_request in (
            ("batch110", HYBRID_IMAGE_BATCH_CONTROL_REQUEST),
            ("batch200", HYBRID_IMAGE_BATCH_CANDIDATE_REQUEST),
        ):
            warm = _run_transition_arm(
                checkpoint,
                label=f"prewarm_{batch_label}",
                candidate_mode=args.candidate_mode,
                candidate_enabled=True,
                checkpoint_iteration=args.checkpoint_iteration,
                hybrid_image_batch_request=batch_request,
            )
            for key, expected in expected_manifests.items():
                if warm[key]["manifest_sha256"] != expected:
                    raise RuntimeError(
                        f"prewarm_{batch_label} did not start from the exact shared {key}",
                    )
            batch_contract = warm["execution_contract"]["hybrid_image_batch"]
            prewarm[batch_label] = {
                "label": warm["label"],
                "backend_mode": warm["backend_mode"],
                "wall_s": warm["wall_s"],
                "hybrid_image_batch": batch_contract,
                "initial_state_manifest_sha256": warm["initial_state_manifest"][
                    "manifest_sha256"
                ],
                "initial_particle_state_manifest_sha256": warm[
                    "initial_particle_state_manifest"
                ]["manifest_sha256"],
                "initial_sampling_state_manifest_sha256": warm[
                    "initial_sampling_state_manifest"
                ]["manifest_sha256"],
            }
            del warm
            gc.collect()
        arm_specs = _hybrid_image_batch_arm_specs()
    elif args.candidate_mode == "exact_coarse_single_translate":
        for backend, enabled in (("skip_off", False), ("skip_on", True)):
            warm = _run_transition_arm(
                checkpoint,
                label=f"prewarm_{backend}",
                candidate_mode=args.candidate_mode,
                candidate_enabled=enabled,
                checkpoint_iteration=args.checkpoint_iteration,
                exact_coarse_profile_enabled=True,
            )
            for key, expected in expected_manifests.items():
                if warm[key]["manifest_sha256"] != expected:
                    raise RuntimeError(
                        f"prewarm_{backend} did not start from the exact shared {key}",
                    )
            prewarm[backend] = {
                "label": warm["label"],
                "backend_mode": warm["backend_mode"],
                "wall_s": warm["wall_s"],
                "exact_coarse_single_translate": warm["execution_contract"][
                    "exact_coarse_single_translate"
                ],
                "initial_state_manifest_sha256": warm[
                    "initial_state_manifest"
                ]["manifest_sha256"],
                "initial_particle_state_manifest_sha256": warm[
                    "initial_particle_state_manifest"
                ]["manifest_sha256"],
                "initial_sampling_state_manifest_sha256": warm[
                    "initial_sampling_state_manifest"
                ]["manifest_sha256"],
            }
            del warm
            gc.collect()
        arm_specs = tuple(
            (label, None)
            for label in EXACT_COARSE_SINGLE_TRANSLATE_ARM_ORDER
        )
    elif args.candidate_mode == "exact_compact_preprocess":
        for backend, enabled in (
            ("preprocess_off", False),
            ("preprocess_on", True),
        ):
            warm = _run_transition_arm(
                checkpoint,
                label=f"prewarm_{backend}",
                candidate_mode=args.candidate_mode,
                candidate_enabled=enabled,
                checkpoint_iteration=args.checkpoint_iteration,
                exact_coarse_profile_enabled=True,
            )
            for key, expected in expected_manifests.items():
                if warm[key]["manifest_sha256"] != expected:
                    raise RuntimeError(
                        f"prewarm_{backend} did not start from the exact shared {key}",
                    )
            prewarm[backend] = {
                "label": warm["label"],
                "backend_mode": warm["backend_mode"],
                "wall_s": warm["wall_s"],
                "exact_compact_preprocess": warm["execution_contract"][
                    "exact_compact_preprocess"
                ],
                "initial_state_manifest_sha256": warm[
                    "initial_state_manifest"
                ]["manifest_sha256"],
                "initial_particle_state_manifest_sha256": warm[
                    "initial_particle_state_manifest"
                ]["manifest_sha256"],
                "initial_sampling_state_manifest_sha256": warm[
                    "initial_sampling_state_manifest"
                ]["manifest_sha256"],
            }
            del warm
            gc.collect()
        arm_specs = tuple(
            (label, None)
            for label in EXACT_COMPACT_PREPROCESS_ARM_ORDER
        )
    elif args.candidate_mode == "fused_pair_fine_score":
        for backend, enabled in (("pair_off", False), ("pair_on", True)):
            warm = _run_transition_arm(
                checkpoint,
                label=f"prewarm_{backend}",
                candidate_mode=args.candidate_mode,
                candidate_enabled=enabled,
                checkpoint_iteration=args.checkpoint_iteration,
            )
            for key, expected in expected_manifests.items():
                if warm[key]["manifest_sha256"] != expected:
                    raise RuntimeError(
                        f"prewarm_{backend} did not start from the exact shared {key}",
                    )
            prewarm[backend] = {
                "label": warm["label"],
                "backend_mode": warm["backend_mode"],
                "wall_s": warm["wall_s"],
                "performance_summary": warm["performance_summary"],
                "persistent_cache": warm["persistent_cache"],
                "fused_pair_fine_score": warm["execution_contract"][
                    "fused_pair_fine_score"
                ],
                "initial_state_manifest_sha256": warm[
                    "initial_state_manifest"
                ]["manifest_sha256"],
                "initial_particle_state_manifest_sha256": warm[
                    "initial_particle_state_manifest"
                ]["manifest_sha256"],
                "initial_sampling_state_manifest_sha256": warm[
                    "initial_sampling_state_manifest"
                ]["manifest_sha256"],
            }
            del warm
            gc.collect()
        arm_specs = tuple(
            (label, None)
            for label in FUSED_PAIR_FINE_SCORE_ARM_ORDER
        )
    elif args.candidate_mode == "fused_coarse_projector":
        for backend, enabled in (("fused_off", False), ("fused_on", True)):
            warm = _run_transition_arm(
                checkpoint,
                label=f"prewarm_{backend}",
                candidate_mode=args.candidate_mode,
                candidate_enabled=enabled,
                checkpoint_iteration=args.checkpoint_iteration,
            )
            for key, expected in expected_manifests.items():
                if warm[key]["manifest_sha256"] != expected:
                    raise RuntimeError(
                        f"prewarm_{backend} did not start from the exact shared {key}",
                    )
            prewarm[backend] = {
                "label": warm["label"],
                "backend_mode": warm["backend_mode"],
                "wall_s": warm["wall_s"],
                "persistent_cache": warm["persistent_cache"],
                "fused_coarse_projector": warm["execution_contract"][
                    "fused_coarse_projector"
                ],
                "initial_state_manifest_sha256": warm[
                    "initial_state_manifest"
                ]["manifest_sha256"],
                "initial_particle_state_manifest_sha256": warm[
                    "initial_particle_state_manifest"
                ]["manifest_sha256"],
                "initial_sampling_state_manifest_sha256": warm[
                    "initial_sampling_state_manifest"
                ]["manifest_sha256"],
            }
            del warm
            gc.collect()
        arm_specs = tuple(
            (label, None)
            for label in FUSED_COARSE_PROJECTOR_ARM_ORDER
        )
    else:
        legacy_arm_order = _arm_order(args.candidate_mode)
        arm_specs = tuple(
            (label, None)
            for label in legacy_arm_order
        )
    arm_order = tuple(label for label, _arm_parameter in arm_specs)
    arms: dict[str, dict[str, Any]] = {}
    for label, arm_parameter in arm_specs:
        if args.mirrored_hybrid_image_batch_panels:
            backend_mode = None
            hybrid_image_batch_request = arm_parameter
            candidate_enabled = hybrid_image_batch_request is not None
        else:
            backend_mode = arm_parameter
            hybrid_image_batch_request = None
            candidate_enabled = (
                _arm_candidate_enabled(label, args.candidate_mode)
                if backend_mode is None
                else backend_mode == args.candidate_mode
            )
        arms[label] = _run_transition_arm(
            checkpoint,
            label=label,
            candidate_mode=args.candidate_mode,
            candidate_enabled=candidate_enabled,
            checkpoint_iteration=args.checkpoint_iteration,
            backend_mode=backend_mode,
            hybrid_image_batch_request=hybrid_image_batch_request,
            fused_posterior_dump_original_index=(
                args.fused_posterior_dump_original_index
            ),
            coarse_prefix_dump_original_index=(
                args.coarse_prefix_dump_original_index
            ),
        )

    for label, arm in arms.items():
        for key, expected in expected_manifests.items():
            if arm[key]["manifest_sha256"] != expected:
                raise RuntimeError(f"{label} did not start from the exact shared {key}")

    if args.mirrored_incremental_panels:
        pair_labels = _incremental_pair_labels()
    elif args.mirrored_hybrid_image_batch_panels:
        pair_labels = _hybrid_image_batch_pair_labels()
    else:
        candidate_1, candidate_2 = arm_order[1:3]
        pair_labels = (
            (arm_order[0], arm_order[3]),
            (candidate_1, candidate_2),
            (arm_order[0], candidate_1),
            (arm_order[0], candidate_2),
            (arm_order[3], candidate_1),
            (arm_order[3], candidate_2),
        )
    comparisons = {
        f"{left}__vs__{right}": _pair_report(arms[left], arms[right])
        for left, right in pair_labels
    }
    hybrid_image_batch_science_contract = (
        _hybrid_image_batch_science_contract(comparisons)
        if args.mirrored_hybrid_image_batch_panels
        else None
    )
    hybrid_image_batch_runtime_contract = (
        _hybrid_image_batch_runtime_contract(arms)
        if args.mirrored_hybrid_image_batch_panels
        else None
    )
    compact_science_contract = None
    if (
        not args.mirrored_hybrid_image_batch_panels
        and _candidate_uses_compact_posterior(args.candidate_mode)
    ):
        compact_science_contract = _compact_science_contract(comparisons, arm_order)
    exact_coarse_single_translate_runtime_contract = (
        _exact_coarse_single_translate_runtime_contract(arms)
        if args.candidate_mode == "exact_coarse_single_translate"
        else None
    )
    exact_coarse_single_translate_atomic_contract_passed = (
        bool(
            compact_science_contract["accumulator_repeat_envelope"][
                "all_cross_within_observed_repeat_envelope"
            ]
            and compact_science_contract["final_state_repeat_envelope"][
                "all_cross_within_observed_repeat_envelope"
            ]
        )
        if args.candidate_mode == "exact_coarse_single_translate"
        else None
    )
    exact_compact_preprocess_runtime_contract = (
        _exact_compact_preprocess_runtime_contract(arms)
        if args.candidate_mode == "exact_compact_preprocess"
        else None
    )
    exact_compact_preprocess_atomic_contract_passed = (
        bool(
            compact_science_contract["accumulator_repeat_envelope"][
                "all_cross_within_observed_repeat_envelope"
            ]
            and compact_science_contract["final_state_repeat_envelope"][
                "all_cross_within_observed_repeat_envelope"
            ]
        )
        if args.candidate_mode == "exact_compact_preprocess"
        else None
    )
    fused_pair_fine_runtime_contract = (
        _fused_pair_fine_runtime_contract(arms)
        if args.candidate_mode == "fused_pair_fine_score"
        else None
    )
    fused_pair_fine_atomic_contract_passed = (
        bool(compact_science_contract["atomic_repeat_envelope_passed"])
        if args.candidate_mode == "fused_pair_fine_score"
        else None
    )
    gpu_memory = _gpu_memory_report(arms)
    target_arm_summaries: dict[str, Any] | None = None
    if target_row_index is not None:
        target_arm_summaries = {
            label: _target_particle_summary(
                arm["particle_state"],
                arm["estep_meta"],
                row_index=target_row_index,
                pixel_size=float(checkpoint["dataset"].voxel_size),
            )
            for label, arm in arms.items()
        }
    arm_dir = output_root / "arms"
    arm_dir.mkdir()
    arm_summaries: dict[str, Any] = {}
    for label, arm in arms.items():
        payload = {
            "label": label,
            "hybrid": arm["hybrid"],
            "candidate_mode": arm["candidate_mode"],
            "candidate_enabled": arm["candidate_enabled"],
            "backend_mode": arm["backend_mode"],
            "execution_contract": arm["execution_contract"],
            "performance_summary": arm["performance_summary"],
            "wall_s": arm["wall_s"],
            "wall_clock_started_epoch_s": arm["wall_clock_started_epoch_s"],
            "wall_clock_ended_epoch_s": arm["wall_clock_ended_epoch_s"],
            "persistent_cache": arm["persistent_cache"],
            "gpu_memory": gpu_memory["arms"].get(label),
            "stable_fourier_window_contract": arm[
                "stable_fourier_window_contract"
            ],
            "stable_flat_capacity_contract": arm[
                "stable_flat_capacity_contract"
            ],
            "initial_state_manifest": arm["initial_state_manifest"],
            "initial_particle_state_manifest": arm["initial_particle_state_manifest"],
            "initial_sampling_state_manifest": arm["initial_sampling_state_manifest"],
            "accumulator_manifest": _accumulator_manifest(arm["accumulators"]),
            "final_state_manifest": _dataclass_manifest(arm["final_state"]),
            "particle_state_manifest": _dataclass_manifest(arm["particle_state"]),
            "sampling_state_manifest": _dataclass_manifest(arm["sampling_state"]),
            "estep_meta": _json_ready(arm["estep_meta"]),
            "post_iteration_meta": _json_ready(arm["post_iteration_meta"]),
            "target_particle": (
                None
                if target_arm_summaries is None
                else target_arm_summaries[label]
            ),
        }
        arm_path = arm_dir / f"{label}.json"
        arm_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        arm_summaries[label] = {
            "hybrid": arm["hybrid"],
            "candidate_mode": arm["candidate_mode"],
            "candidate_enabled": arm["candidate_enabled"],
            "backend_mode": arm["backend_mode"],
            "execution_contract": arm["execution_contract"],
            "performance_summary": arm["performance_summary"],
            "wall_s": arm["wall_s"],
            "wall_clock_started_epoch_s": arm["wall_clock_started_epoch_s"],
            "wall_clock_ended_epoch_s": arm["wall_clock_ended_epoch_s"],
            "persistent_cache": arm["persistent_cache"],
            "gpu_memory": payload["gpu_memory"],
            "stable_fourier_window_contract": payload[
                "stable_fourier_window_contract"
            ],
            "stable_flat_capacity_contract": payload[
                "stable_flat_capacity_contract"
            ],
            "artifact": str(arm_path.resolve()),
            "artifact_sha256": _sha256_bytes(arm_path.read_bytes()),
            "accumulator_manifest": payload["accumulator_manifest"],
            "final_state_manifest": payload["final_state_manifest"],
            "particle_state_manifest": payload["particle_state_manifest"],
            "sampling_state_manifest": payload["sampling_state_manifest"],
            "target_particle": payload["target_particle"],
        }

    report = {
        "schema": SCHEMA,
        "classification": "diagnostic_same_in_memory_state_one_transition_only",
        "checkpoint_iteration": int(args.checkpoint_iteration),
        "profiled_iteration": int(args.checkpoint_iteration) + 1,
        "candidate_mode": args.candidate_mode,
        "checkpoint_source": checkpoint["checkpoint_source"],
        "target_particle": (
            None
            if target_checkpoint is None
            else {
                **target_checkpoint,
                "arms": target_arm_summaries,
            }
        ),
        "fused_posterior_dump_original_index": (
            args.fused_posterior_dump_original_index
        ),
        "coarse_prefix_dump_original_index": (
            args.coarse_prefix_dump_original_index
        ),
        "coarse_prefix_dump_dir": (
            str((output_root / "coarse_prefix_dumps").resolve())
            if args.coarse_prefix_dump_original_index is not None
            else None
        ),
        "fused_posterior_dump_dir": (
            str((output_root / "fused_posterior_dumps").resolve())
            if args.fused_posterior_dump_original_index is not None
            else None
        ),
        "mirrored_incremental_panels": bool(args.mirrored_incremental_panels),
        "mirrored_hybrid_image_batch_panels": bool(
            args.mirrored_hybrid_image_batch_panels,
        ),
        "frozen_nr_iter_schedule": frozen_nr_iter,
        "arm_order": list(arm_order),
        "panel_orders": (
            {
                "abba": list(PACKED_FINAL_NOISE_INCREMENTAL_ABBA_ARM_ORDER),
                "baab": list(PACKED_FINAL_NOISE_INCREMENTAL_BAAB_ARM_ORDER),
            }
            if args.mirrored_incremental_panels
            else {
                "abba": list(HYBRID_IMAGE_BATCH_ABBA_ARM_ORDER),
                "baab": list(HYBRID_IMAGE_BATCH_BAAB_ARM_ORDER),
            }
            if args.mirrored_hybrid_image_batch_panels
            else {"abba": list(arm_order)}
        ),
        "prewarm": prewarm,
        "checkpoint_wall_s": checkpoint_wall_s,
        "checkpoint_manifest": checkpoint_manifest,
        "checkpoint_execution_contract": checkpoint[
            "checkpoint_execution_contract"
        ],
        "fixture_dir": str(fixture_dir),
        "acceptance_config": str(acceptance_path),
        "acceptance_config_sha256": _sha256_bytes(acceptance_path.read_bytes()),
        "arms": arm_summaries,
        "comparisons": comparisons,
        "compact_science_contract": compact_science_contract,
        "exact_coarse_single_translate_runtime_contract": (
            exact_coarse_single_translate_runtime_contract
        ),
        "exact_coarse_single_translate_atomic_contract_passed": (
            exact_coarse_single_translate_atomic_contract_passed
        ),
        "exact_compact_preprocess_runtime_contract": (
            exact_compact_preprocess_runtime_contract
        ),
        "exact_compact_preprocess_atomic_contract_passed": (
            exact_compact_preprocess_atomic_contract_passed
        ),
        "fused_pair_fine_runtime_contract": fused_pair_fine_runtime_contract,
        "fused_pair_fine_atomic_contract_passed": (
            fused_pair_fine_atomic_contract_passed
        ),
        "gpu_memory": gpu_memory,
        "hybrid_image_batch_science_contract": (
            hybrid_image_batch_science_contract
        ),
        "hybrid_image_batch_runtime_contract": (
            hybrid_image_batch_runtime_contract
        ),
        "same_state_contract": {
            "model_state_exact_for_every_arm": True,
            "particle_state_exact_for_every_arm": True,
            "sampling_state_exact_for_every_arm": True,
            "checkpoint_trajectory_backend": (
                "native_relion_load_only"
                if native_checkpoint is not None
                else "all_optimized_stable_shapes_on_q32_batched"
                if args.candidate_mode == "all_optimized_stable_shapes"
                else "direct"
            ),
            "only_transition_differences": (
                ["stable_fourier_window_shapes", "stable_flat_row_capacity"]
                if args.candidate_mode == "all_optimized_stable_shapes"
                else [FUSED_COARSE_PROJECTOR_ENVIRONMENT]
                if args.candidate_mode == "fused_coarse_projector"
                else None
            ),
            "baseline_trajectory_backend": (
                "packed_deferred"
                if args.mirrored_incremental_panels
                else "all_optimized_hybrid_image_batch_110"
                if args.mirrored_hybrid_image_batch_panels
                else "all_optimized_exact_coarse_single_translate_off"
                if args.candidate_mode == "exact_coarse_single_translate"
                else "all_optimized_exact_compact_preprocess_off"
                if args.candidate_mode == "exact_compact_preprocess"
                else "all_optimized_fused_pair_fine_score_off"
                if args.candidate_mode == "fused_pair_fine_score"
                else "rectangular_preprojected_coarse_scorer"
                if args.candidate_mode == "fused_coarse_projector"
                else "all_optimized_stable_shapes_off_q32_batched"
                if args.candidate_mode == "all_optimized_stable_shapes"
                else (
                    "hybrid_packed_deferred_stable_fourier_off"
                    if args.candidate_mode == "stable_shapes"
                    else (
                        "hybrid_packed_deferred_stable_flat_capacity_off"
                        if args.candidate_mode == "stable_flat_capacity"
                        else "direct"
                    )
                )
            ),
            "candidate_backend": (
                "all_optimized_hybrid_image_batch_200"
                if args.mirrored_hybrid_image_batch_panels
                else "all_optimized_exact_coarse_single_translate_on"
                if args.candidate_mode == "exact_coarse_single_translate"
                else "all_optimized_exact_compact_preprocess_on"
                if args.candidate_mode == "exact_compact_preprocess"
                else "all_optimized_fused_pair_fine_score_on"
                if args.candidate_mode == "fused_pair_fine_score"
                else "shared_relion_fused_coarse_projector"
                if args.candidate_mode == "fused_coarse_projector"
                else "all_optimized_stable_shapes_on_q32_batched"
                if args.candidate_mode == "all_optimized_stable_shapes"
                else (
                    "hybrid_packed_deferred_stable_fourier_on"
                    if args.candidate_mode == "stable_shapes"
                    else (
                        "hybrid_packed_deferred_stable_flat_capacity_on"
                        if args.candidate_mode == "stable_flat_capacity"
                        else args.candidate_mode
                    )
                )
            ),
            "transition_panel": (
                "packed_deferred/packed_final_noise/packed_final_noise/packed_deferred"
                "+packed_final_noise/packed_deferred/packed_deferred/packed_final_noise"
                if args.mirrored_incremental_panels
                else "batch110/batch200/batch200/batch110"
                "+batch200/batch110/batch110/batch200"
                if args.mirrored_hybrid_image_batch_panels
                else "/".join(arm_order)
            ),
            "both_compared_backends_prewarmed": bool(
                args.mirrored_incremental_panels
                or args.mirrored_hybrid_image_batch_panels
                or args.candidate_mode
                in {
                    "exact_coarse_single_translate",
                    "exact_compact_preprocess",
                    "fused_pair_fine_score",
                    "fused_coarse_projector",
                }
            ),
            "both_incremental_backends_prewarmed": bool(
                args.mirrored_incremental_panels,
            ),
            "direct_oracle_from_shared_checkpoint": bool(
                args.mirrored_incremental_panels
                or args.mirrored_hybrid_image_batch_panels
            ),
            "oracle_backend": "direct",
            "support_audit_ids_enabled_for_transition_arms": True,
            "requested_environment_exact_for_every_arm": True,
            "compact_profile_fail_closed": _candidate_uses_compact_posterior(
                args.candidate_mode,
            ),
            "all_optimized_profile_fail_closed": args.candidate_mode
            in {
                "all_optimized",
                "exact_coarse_single_translate",
                "exact_compact_preprocess",
                "fused_pair_fine_score",
            },
            "exact_coarse_single_translate_profile_fail_closed": (
                args.candidate_mode == "exact_coarse_single_translate"
            ),
            "exact_compact_preprocess_profile_fail_closed": (
                args.candidate_mode == "exact_compact_preprocess"
            ),
            "fused_pair_fine_score_profile_fail_closed": (
                args.candidate_mode == "fused_pair_fine_score"
            ),
            "fused_coarse_projector_profile_fail_closed": (
                args.candidate_mode == "fused_coarse_projector"
            ),
            "persistent_cache_family_counts_recorded": (
                args.candidate_mode == "fused_pair_fine_score"
            ),
            "gpu_hbm_samples_recorded": bool(
                args.candidate_mode == "fused_pair_fine_score"
                and gpu_memory["available"]
                and gpu_memory["all_arms_sampled"]
            ),
            "hybrid_image_batch_profile_fail_closed": bool(
                args.mirrored_hybrid_image_batch_panels,
            ),
        },
        "science_divergence_is_observational_not_exit_gate": bool(
            native_checkpoint is not None
            or args.candidate_mode == "fused_coarse_projector"
        ),
        "science_promotion_allowed": False,
    }
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": str(report_path.resolve()),
                "report_sha256": _sha256_bytes(report_path.read_bytes()),
                "checkpoint_wall_s": checkpoint_wall_s,
                "arm_walls_s": {label: arms[label]["wall_s"] for label in arm_order},
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    enforce_science_exit_gate = not bool(
        native_checkpoint is not None
        or args.candidate_mode == "fused_coarse_projector"
    )
    if enforce_science_exit_gate:
        if (
            compact_science_contract is not None
            and not compact_science_contract["hard_exact_contract_passed"]
        ):
            return 1
        if hybrid_image_batch_science_contract is not None and not (
            hybrid_image_batch_science_contract["hard_exact_contract_passed"]
            and hybrid_image_batch_science_contract["atomic_repeat_envelope_passed"]
        ):
            return 1
        if args.candidate_mode == "exact_coarse_single_translate" and not (
            compact_science_contract["hard_exact_contract_passed"]
            and exact_coarse_single_translate_atomic_contract_passed
        ):
            return 1
        if args.candidate_mode == "exact_compact_preprocess" and not (
            compact_science_contract["hard_exact_contract_passed"]
            and exact_compact_preprocess_atomic_contract_passed
        ):
            return 1
        if args.candidate_mode == "fused_pair_fine_score" and not (
            compact_science_contract["hard_exact_contract_passed"]
            and fused_pair_fine_atomic_contract_passed
            and fused_pair_fine_runtime_contract["persistent_cache_contract"][
                "timed_arms_add_no_target_family_programs_after_prewarm"
            ]
            and fused_pair_fine_runtime_contract["material_speedup_passed"]
            and gpu_memory["available"]
            and gpu_memory["all_arms_sampled"]
        ):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
