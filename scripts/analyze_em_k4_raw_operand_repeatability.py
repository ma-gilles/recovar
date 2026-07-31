#!/usr/bin/env python3
"""Audit same-allocation K=4 raw-score operand repeatability."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

OPERAND_FIELDS = (
    ("shifted_image", "raw_operand_shifted_corrected"),
    ("projection", "raw_operand_proj_half"),
    ("score_weight", "raw_operand_corr_img_score"),
    ("half_weight", "raw_operand_half_weights"),
    ("high_shell_scalar", "raw_operand_highres_xi2_half"),
)
TOPOLOGY_FIELDS = (
    "candidate_mask",
    "rotations",
    "oversampled_rot_indices",
    "parent_map",
    "fine_translations",
    "fine_translation_parent",
    "raw_operand_relion_full_to_compact",
)
REQUIRED_FIELDS = {
    "relion_raw_diff2",
    "original_index",
    "class_index",
    "current_size",
    *TOPOLOGY_FIELDS,
    *(field for _, field in OPERAND_FIELDS),
}


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        values = {key: np.asarray(archive[key]) for key in archive.files}
    missing = sorted(REQUIRED_FIELDS - values.keys())
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")
    return values


def _strict_element_stats(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    left = np.asarray(left)
    right = np.asarray(right)
    shape_equal = left.shape == right.shape
    dtype_equal = left.dtype == right.dtype
    if not shape_equal or not dtype_equal:
        return {
            "shape_equal": shape_equal,
            "dtype_equal": dtype_equal,
            "element_count": int(left.size),
            "mismatch_count": None,
            "byte_equal": False,
        }
    left_bytes = np.ascontiguousarray(left).reshape(-1).view(np.uint8).reshape(-1, left.dtype.itemsize)
    right_bytes = np.ascontiguousarray(right).reshape(-1).view(np.uint8).reshape(-1, right.dtype.itemsize)
    mismatch = np.any(left_bytes != right_bytes, axis=1)
    return {
        "shape_equal": True,
        "dtype_equal": True,
        "element_count": int(left.size),
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "byte_equal": not bool(np.any(mismatch)),
    }


def _raw_stats(
    left: np.ndarray,
    right: np.ndarray,
    candidate_mask: np.ndarray,
) -> dict[str, object]:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if left.shape != right.shape or left.shape != candidate_mask.shape:
        raise ValueError(
            "raw replay, reference, and candidate-mask shapes must agree: "
            f"{left.shape}, {right.shape}, {candidate_mask.shape}"
        )
    left_active = left[candidate_mask]
    right_active = right[candidate_mask]
    mismatch = left_active.view(np.uint32) != right_active.view(np.uint32)
    finite = np.isfinite(left_active) & np.isfinite(right_active)
    max_abs = (
        float(
            np.max(
                np.abs(
                    left_active[finite].astype(np.float64)
                    - right_active[finite].astype(np.float64)
                )
            )
        )
        if np.any(finite)
        else None
    )
    return {
        "active_count": int(left_active.size),
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "byte_equal": not bool(np.any(mismatch)),
        "max_abs_difference": max_abs,
    }


def _jax_replay(values: dict[str, np.ndarray]) -> np.ndarray:
    import jax.numpy as jnp

    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _score_pass2_bucket_relion_gpu_diff2_raw,
    )

    replay = _score_pass2_bucket_relion_gpu_diff2_raw(
        jnp.asarray(values["raw_operand_shifted_corrected"])[None, ...],
        jnp.asarray(values["raw_operand_corr_img_score"])[None, ...],
        jnp.asarray(values["raw_operand_proj_half"])[None, ...],
        jnp.asarray(values["raw_operand_half_weights"]),
        jnp.asarray(values["raw_operand_relion_full_to_compact"]),
        jnp.asarray(values["raw_operand_highres_xi2_half"])[None],
    )
    return np.asarray(replay[0], dtype=np.float32)


def analyze(
    first_path: Path,
    second_path: Path,
    *,
    replay_fn: Callable[[dict[str, np.ndarray]], np.ndarray] = _jax_replay,
) -> dict[str, object]:
    first = _load_archive(first_path)
    second = _load_archive(second_path)

    topology = {
        field: _strict_element_stats(first[field], second[field])
        for field in TOPOLOGY_FIELDS
    }
    topology_equal = all(record["byte_equal"] for record in topology.values())
    identity_equal = all(
        np.array_equal(first[field], second[field])
        for field in ("original_index", "class_index", "current_size")
    )
    operands = {
        family: _strict_element_stats(first[field], second[field])
        for family, field in OPERAND_FIELDS
    }

    first_replay = replay_fn(first)
    second_replay = replay_fn(second)
    first_self = _raw_stats(
        first_replay,
        first["relion_raw_diff2"],
        first["candidate_mask"],
    )
    second_self = _raw_stats(
        second_replay,
        second["relion_raw_diff2"],
        second["candidate_mask"],
    )
    self_replay_passed = bool(first_self["byte_equal"] and second_self["byte_equal"])

    baseline = _raw_stats(
        first_replay,
        second["relion_raw_diff2"],
        second["candidate_mask"],
    )
    substitutions = {}
    differing_families = [
        family for family, _ in OPERAND_FIELDS if not operands[family]["byte_equal"]
    ]
    if identity_equal and topology_equal and self_replay_passed:
        for family, field in OPERAND_FIELDS:
            if family not in differing_families:
                continue
            substituted = dict(first)
            substituted[field] = second[field]
            stats = _raw_stats(
                replay_fn(substituted),
                second["relion_raw_diff2"],
                second["candidate_mask"],
            )
            stats["mismatch_reduction"] = (
                int(baseline["mismatch_count"]) - int(stats["mismatch_count"])
            )
            substitutions[family] = stats

    maximizers: list[str] = []
    if substitutions:
        maximum = max(int(record["mismatch_reduction"]) for record in substitutions.values())
        maximizers = [
            family
            for family, record in substitutions.items()
            if int(record["mismatch_reduction"]) == maximum
        ]

    if not identity_equal or not topology_equal:
        classification = "identity_or_topology_mismatch"
    elif not self_replay_passed:
        classification = "captured_operands_do_not_self_replay"
    elif not differing_families and baseline["byte_equal"]:
        classification = "effective_operands_and_raw_costs_repeat_bit_for_bit"
    elif not differing_families:
        classification = "raw_costs_differ_without_captured_operand_difference"
    elif len(differing_families) == 1:
        classification = f"first_captured_divergence_{differing_families[0]}"
    elif len(maximizers) == 1:
        classification = f"multiple_operands_differ_unique_substitution_{maximizers[0]}"
    else:
        classification = "multiple_operands_differ_attribution_tie"

    return {
        "schema": "recovar.em_k4_raw_operand_repeatability.v1",
        "status": "complete",
        "classification": classification,
        "identity_equal": identity_equal,
        "topology_equal": topology_equal,
        "self_replay_passed": self_replay_passed,
        "repeatable": bool(
            identity_equal
            and topology_equal
            and self_replay_passed
            and not differing_families
            and baseline["byte_equal"]
        ),
        "topology": topology,
        "operands": operands,
        "differing_operand_families": differing_families,
        "self_replay": {
            "first": first_self,
            "second": second_self,
        },
        "cross_arm_raw": baseline,
        "single_family_substitutions": substitutions,
        "maximum_mismatch_reduction_families": maximizers,
        "scorecard_change_admissible": False,
        "correlation_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = analyze(args.first, args.second)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
