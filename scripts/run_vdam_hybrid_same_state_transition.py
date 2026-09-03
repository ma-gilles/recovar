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
import dataclasses
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
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--acceptance-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-iteration", type=int, default=34)
    parser.add_argument("--image-batch-size", type=int, default=500)
    parser.add_argument("--candidate-mode", choices=CANDIDATE_MODES, default="hybrid")
    parser.add_argument(
        "--mirrored-incremental-panels",
        action="store_true",
        help=(
            "prewarm packed-deferred and packed-final, then measure mirrored "
            "ABBA/BAAB panels between only those two backends"
        ),
    )
    return parser.parse_args(argv)


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
    }
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
    }


def _candidate_uses_compact_posterior(candidate_mode: str) -> bool:
    return candidate_mode in {
        "compact_posterior",
        "compact_packed_deferred",
        "all_optimized",
    }


def _candidate_uses_packed_deferred(candidate_mode: str) -> bool:
    return candidate_mode in {
        "compact_packed_deferred",
        "all_optimized",
    }


def _arm_candidate_enabled(label: str, candidate_mode: str) -> bool:
    if candidate_mode == "stable_shapes":
        return label.startswith("stable_on_")
    if candidate_mode == "stable_flat_capacity":
        return label.startswith("stable_flat_on_")
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
) -> dict[str, Any]:
    """Prove logical Fourier support was carried by the requested capacity."""

    from recovar.em.dense_single_volume.helpers.fourier_window import (
        make_stable_fourier_window_shape_plan,
    )

    image_shape = tuple(int(value) for value in image_shape)
    if len(image_shape) != 2 or image_shape[0] != image_shape[1]:
        raise RuntimeError(f"{label} has invalid image shape {image_shape}")
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
            recon_exact_radius=False,
        )
        expected = {
            "stable_fourier_window_shapes": bool(
                enabled and plan.logical_spec.use_window
            ),
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
    "stable_flat_row_capacity",
)


def _validate_all_optimized_profiles(
    estep_meta: dict[str, Any],
    *,
    enabled: bool,
    label: str,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    """Fail closed unless every seam in the composed arm is effective."""

    stable_fourier = _validate_stable_fourier_profiles(
        estep_meta,
        enabled=enabled,
        label=label,
        image_shape=image_shape,
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
    return {
        "enabled": bool(enabled),
        "enabled_seams": list(ALL_OPTIMIZED_SEAMS if enabled else ()),
        "disabled_seams": list(() if enabled else ALL_OPTIMIZED_SEAMS),
        "profile_exact": True,
        "stable_fourier": stable_fourier,
        "stable_flat_capacity": stable_flat,
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


def _validate_arm_execution_contract(
    *,
    candidate_mode: str,
    candidate_enabled: bool,
    requested_environment: dict[str, str],
    effective_environment: dict[str, str | None],
    estep_meta: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed when a compact arm does not execute the requested profile."""

    if effective_environment != requested_environment:
        raise RuntimeError(
            "same-state arm environment differs from its explicit request: "
            f"requested={requested_environment}, effective={effective_environment}",
        )
    contract: dict[str, Any] = {
        "requested_environment": dict(requested_environment),
        "effective_environment": dict(effective_environment),
        "environment_exact": True,
        "profile_checked": _candidate_uses_compact_posterior(candidate_mode),
    }
    if not _candidate_uses_compact_posterior(candidate_mode):
        return contract

    profiles = _coarse_hybrid_profiles(estep_meta)
    contract["hybrid_profiles"] = _json_ready(profiles)
    expected_compact = bool(candidate_enabled)
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
            "selected_rescore_batch_count",
            "selected_rescore_image_count",
            "selected_source16_block_count",
            "selected_exact_candidate_count",
            "selected_score_table_capacity_candidates",
            "dense_global_score_table_capacity_candidates",
            "selected_score_table_capacity_bytes_f32",
            "dense_global_score_table_capacity_bytes_f32",
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
        if profile["selected_rescore_batch_count"] != profile["batch_count"]:
            raise RuntimeError(
                f"{profile_name} compact profile did not select every batch",
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


def _arm_performance_summary(
    *,
    wall_s: float,
    estep_meta: dict[str, Any],
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
    table_fields = (
        "selected_rescore_batch_count",
        "fallback_batch_count",
        "selected_rescore_image_count",
        "selected_source16_block_count",
        "selected_exact_candidate_count",
        "full_candidate_count_for_selected_images",
        "selected_exact_candidate_fraction",
        "selected_score_table_capacity_candidates",
        "dense_global_score_table_capacity_candidates",
        "selected_score_table_capacity_bytes_f32",
        "dense_global_score_table_capacity_bytes_f32",
        "selected_to_dense_score_table_capacity_fraction",
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
    if "packed_final_noise" in label:
        return "packed_final_noise"
    if "packed_deferred" in label:
        return "packed_deferred"
    raise ValueError(f"unsupported incremental arm label: {label}")


def _incremental_arm_specs() -> tuple[tuple[str, str], ...]:
    return tuple(
        (label, _incremental_backend_for_label(label))
        for label in PACKED_FINAL_NOISE_INCREMENTAL_ARM_ORDER
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
        ]
    )
    return tuple(pairs)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
) -> dict[str, Any]:
    import recovar.em.initial_model.driver as driver
    from scripts import run_ab_initio
    from scripts.run_vdam_relion_parity_case import build_recovar_command

    input_star = fixture_dir / "particles.star"
    definition = acceptance["science_contract"]["definition"]
    command = build_recovar_command(
        input_star=input_star,
        output_prefix=output_root / "checkpoint" / "run",
        fixture_dir=fixture_dir,
        definition=definition,
        image_batch_size=image_batch_size,
    )
    argv = list(command[3:])
    write_index = argv.index("--grad_write_iter") + 1
    argv[write_index] = str(checkpoint_iteration)
    argv.extend(("--diagnostic_stop_after_iteration", str(checkpoint_iteration)))

    captured: dict[str, Any] = {"argv": argv}
    original_expectation_factory = driver._native_expectation_step
    original_run = driver.run_native_initial_model

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

    driver._native_expectation_step = capture_expectation_factory
    driver.run_native_initial_model = capture_run
    try:
        with _temporary_environment(
            {
                **_candidate_environment("hybrid", enabled=False),
                "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT": "0",
                "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS": "0",
            }
        ):
            status = int(run_ab_initio.main(argv))
    finally:
        driver._native_expectation_step = original_expectation_factory
        driver.run_native_initial_model = original_run
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
) -> dict[str, Any]:
    import recovar.em.initial_model.driver as driver
    from recovar.data_io.starfile import read_star
    from recovar.em.initial_model.schedules import default_subset_sizes_for_3d_initial_model

    state = copy.deepcopy(checkpoint["result"].state)
    particle_state = copy.deepcopy(checkpoint["particle_state"])
    sampling_state = copy.deepcopy(checkpoint["sampling_state"])
    initial_state_manifest = _dataclass_manifest(state)
    initial_particle_state_manifest = _dataclass_manifest(particle_state)
    initial_sampling_state_manifest = _dataclass_manifest(sampling_state)
    opts = checkpoint["opts"]
    if candidate_mode in {"stable_shapes", "all_optimized"}:
        opts = dataclasses.replace(
            opts,
            stable_fourier_window_shapes=bool(candidate_enabled),
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
    resolved_backend_mode = (
        candidate_mode if candidate_enabled else "direct"
    ) if backend_mode is None else backend_mode
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
    effective_environment: dict[str, str | None] = {}
    started = time.perf_counter()
    with _temporary_environment(
        {
            **requested_environment,
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
    if int(final_state.iter) != checkpoint_iteration + 1:
        raise RuntimeError(f"{label} stopped at the wrong iteration")
    if "accumulators" not in captured or "estep_meta" not in captured:
        raise RuntimeError(f"{label} did not capture its E-step boundary")
    stable_fourier_window_contract = None
    if candidate_mode == "stable_shapes":
        stable_fourier_window_contract = _validate_stable_fourier_profiles(
            captured["estep_meta"],
            enabled=candidate_enabled,
            label=label,
            image_shape=tuple(int(value) for value in dataset.image_shape),
        )
    stable_flat_capacity_contract = None
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
    )
    if candidate_mode == "all_optimized":
        all_optimized_contract = _validate_all_optimized_profiles(
            captured["estep_meta"],
            enabled=candidate_enabled,
            label=label,
            image_shape=tuple(int(value) for value in dataset.image_shape),
        )
        execution_contract["all_optimized"] = all_optimized_contract
        execution_contract["all_optimized_profile_exact"] = True
        stable_fourier_window_contract = all_optimized_contract["stable_fourier"]
        if candidate_enabled:
            stable_flat_capacity_contract = all_optimized_contract[
                "stable_flat_capacity"
            ]
    performance_summary = _arm_performance_summary(
        wall_s=wall_s,
        estep_meta=captured["estep_meta"],
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
                    and candidate_mode in {"stable_shapes", "stable_flat_capacity"}
                )
            )
        ),
        "backend_mode": resolved_backend_mode,
        "execution_contract": execution_contract,
        "performance_summary": performance_summary,
        "wall_s": wall_s,
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
    return {
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    fixture_dir = args.fixture_dir.resolve(strict=True)
    acceptance_path = args.acceptance_config.resolve(strict=True)
    output_root = args.output_root.resolve()
    if args.checkpoint_iteration < 1:
        raise ValueError("checkpoint-iteration must be positive")
    if args.image_batch_size < 1:
        raise ValueError("image-batch-size must be positive")
    if args.mirrored_incremental_panels and args.candidate_mode != "packed_final_noise":
        raise ValueError(
            "mirrored incremental panels require --candidate-mode packed_final_noise"
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
    )
    checkpoint_wall_s = float(time.perf_counter() - checkpoint_started)
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
    else:
        legacy_arm_order = _arm_order(args.candidate_mode)
        arm_specs = tuple(
            (label, None)
            for label in legacy_arm_order
        )
    arm_order = tuple(label for label, _backend_mode in arm_specs)
    arms: dict[str, dict[str, Any]] = {}
    for label, backend_mode in arm_specs:
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
        )

    for label, arm in arms.items():
        for key, expected in expected_manifests.items():
            if arm[key]["manifest_sha256"] != expected:
                raise RuntimeError(f"{label} did not start from the exact shared {key}")

    if args.mirrored_incremental_panels:
        pair_labels = _incremental_pair_labels()
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
    compact_science_contract = (
        _compact_science_contract(comparisons, arm_order)
        if _candidate_uses_compact_posterior(args.candidate_mode)
        else None
    )
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
        }

    report = {
        "schema": SCHEMA,
        "classification": "diagnostic_same_in_memory_state_one_transition_only",
        "checkpoint_iteration": int(args.checkpoint_iteration),
        "profiled_iteration": int(args.checkpoint_iteration) + 1,
        "candidate_mode": args.candidate_mode,
        "mirrored_incremental_panels": bool(args.mirrored_incremental_panels),
        "frozen_nr_iter_schedule": frozen_nr_iter,
        "arm_order": list(arm_order),
        "panel_orders": (
            {
                "abba": list(PACKED_FINAL_NOISE_INCREMENTAL_ABBA_ARM_ORDER),
                "baab": list(PACKED_FINAL_NOISE_INCREMENTAL_BAAB_ARM_ORDER),
            }
            if args.mirrored_incremental_panels
            else {"abba": list(arm_order)}
        ),
        "prewarm": prewarm,
        "checkpoint_wall_s": checkpoint_wall_s,
        "checkpoint_manifest": checkpoint_manifest,
        "fixture_dir": str(fixture_dir),
        "acceptance_config": str(acceptance_path),
        "acceptance_config_sha256": _sha256_bytes(acceptance_path.read_bytes()),
        "arms": arm_summaries,
        "comparisons": comparisons,
        "compact_science_contract": compact_science_contract,
        "same_state_contract": {
            "model_state_exact_for_every_arm": True,
            "particle_state_exact_for_every_arm": True,
            "sampling_state_exact_for_every_arm": True,
            "baseline_trajectory_backend": (
                "packed_deferred"
                if args.mirrored_incremental_panels
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
                "hybrid_packed_deferred_stable_fourier_on"
                if args.candidate_mode == "stable_shapes"
                else (
                    "hybrid_packed_deferred_stable_flat_capacity_on"
                    if args.candidate_mode == "stable_flat_capacity"
                    else args.candidate_mode
                )
            ),
            "transition_panel": (
                "packed_deferred/packed_final_noise/packed_final_noise/packed_deferred"
                "+packed_final_noise/packed_deferred/packed_deferred/packed_final_noise"
                if args.mirrored_incremental_panels
                else "/".join(arm_order)
            ),
            "both_incremental_backends_prewarmed": bool(
                args.mirrored_incremental_panels
            ),
            "support_audit_ids_enabled_for_transition_arms": True,
            "requested_environment_exact_for_every_arm": True,
            "compact_profile_fail_closed": _candidate_uses_compact_posterior(
                args.candidate_mode,
            ),
            "all_optimized_profile_fail_closed": args.candidate_mode
            == "all_optimized",
        },
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
    if (
        compact_science_contract is not None
        and not compact_science_contract["hard_exact_contract_passed"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
