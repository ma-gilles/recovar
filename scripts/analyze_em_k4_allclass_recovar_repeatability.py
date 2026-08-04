#!/usr/bin/env python3
"""Audit fixed K=4 all-class RECOVAR pass-2 capture repeatability."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "recovar.em_k4_allclass_recovar_repeatability.v1"
EXPECTED_CLASSES = 4
EXPECTED_ORIGINAL_INDEX = 53_722
EXPECTED_CURRENT_SIZE = 38
GROUP_ORDER = (
    "identity",
    "geometry_and_candidate_tuples",
    "raw_diff2",
    "priors",
    "unnormalized_scores",
    "joint_posterior",
    "global_significant_support",
)
GROUP_FIELDS = {
    "identity": (
        "original_index",
        "local_index",
        "class_index",
        "current_size",
        "n_fine_trans",
        "compact_pair_dump",
    ),
    "geometry_and_candidate_tuples": (
        "fine_translations",
        "fine_translation_parent",
        "rotations",
        "oversampled_rot_indices",
        "parent_map",
        "candidate_mask",
    ),
    "raw_diff2": ("relion_raw_diff2", "relion_min_diff2"),
    "priors": ("rotation_log_prior", "translation_log_prior"),
    "unnormalized_scores": ("scores_pre_prior", "scores_with_prior"),
    "joint_posterior": ("probs",),
    "global_significant_support": (
        "reconstruction_mask",
        "reconstruction_n_significant",
        "reconstruction_probs",
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_metric(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Compare original array bytes and finite numeric values without correlation."""

    left = np.asarray(left)
    right = np.asarray(right)
    _require(left.shape == right.shape, "array shapes differ")
    _require(left.dtype == right.dtype, "array dtypes differ")
    if left.dtype.hasobject:
        raise ValueError("object arrays are not supported")
    byte_rows = left.reshape(-1).view(np.dtype((np.void, left.dtype.itemsize)))
    other_byte_rows = right.reshape(-1).view(np.dtype((np.void, right.dtype.itemsize)))
    mismatch = byte_rows != other_byte_rows
    mismatch_indices = np.flatnonzero(mismatch)
    result: dict[str, Any] = {
        "shape": list(left.shape),
        "dtype": str(left.dtype),
        "byte_exact": bool(not np.any(mismatch)),
        "element_byte_mismatch_count": int(np.count_nonzero(mismatch)),
        "first_mismatch_flat_index": (
            int(mismatch_indices[0]) if mismatch_indices.size else None
        ),
        "correlation_used": False,
    }
    if np.issubdtype(left.dtype, np.number):
        finite = np.isfinite(left) & np.isfinite(right)
        delta = right[finite].astype(np.float64) - left[finite].astype(np.float64)
        result.update(
            {
                "finite_pair_count": int(np.count_nonzero(finite)),
                "same_nan_mask": bool(np.array_equal(np.isnan(left), np.isnan(right))),
                "max_abs_finite_delta": float(np.max(np.abs(delta), initial=0.0)),
                "finite_delta_l2": float(
                    math.sqrt(math.fsum(float(value) * float(value) for value in delta))
                ),
            }
        )
    return result


def classify_first_unequal_group(group_exact: dict[str, bool]) -> str:
    _require(tuple(group_exact) == GROUP_ORDER, "group identity/order changed")
    for group, exact in group_exact.items():
        if not exact:
            return group
    return "all_observed_pass2_fields_exact"


def _load_capture(path: Path, class_one_based: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    expected_fields = {field for fields in GROUP_FIELDS.values() for field in fields}
    _require(expected_fields.issubset(values), f"capture missing {sorted(expected_fields - set(values))}")
    _require(int(values["original_index"]) == EXPECTED_ORIGINAL_INDEX, "particle changed")
    _require(int(values["class_index"]) == class_one_based - 1, "class changed")
    _require(int(values["current_size"]) == EXPECTED_CURRENT_SIZE, "current size changed")
    candidate_mask = np.asarray(values["candidate_mask"], dtype=bool)
    probabilities = np.asarray(values["probs"], dtype=np.float64)
    reconstruction_mask = np.asarray(values["reconstruction_mask"], dtype=bool)
    reconstruction_probabilities = np.asarray(values["reconstruction_probs"], dtype=np.float64)
    _require(
        probabilities.shape == candidate_mask.shape
        and reconstruction_mask.shape == candidate_mask.shape
        and reconstruction_probabilities.shape == candidate_mask.shape,
        "posterior/support shapes changed",
    )
    _require(
        np.all(np.isfinite(probabilities))
        and np.all(probabilities >= 0)
        and np.all(probabilities[~candidate_mask] == 0),
        "joint posterior is invalid",
    )
    _require(
        int(values["reconstruction_n_significant"])
        == int(np.count_nonzero(reconstruction_mask)),
        "significant-support count changed",
    )
    return values


def _arm(root: Path) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    paths = sorted(root.glob("pass2_orig*_class*_cs*.npz"))
    _require(len(paths) == EXPECTED_CLASSES, "arm does not contain four class captures")
    captures = {}
    records = []
    joint_mass = 0.0
    for class_one_based, path in enumerate(paths, 1):
        values = _load_capture(path, class_one_based)
        captures[class_one_based] = values
        mass = math.fsum(float(value) for value in values["probs"].ravel(order="C"))
        joint_mass += mass
        records.append(
            {
                "class_one_based": class_one_based,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "active_candidate_count": int(np.count_nonzero(values["candidate_mask"])),
                "significant_candidate_count": int(np.count_nonzero(values["reconstruction_mask"])),
                "probability_mass": float(mass),
            }
        )
    valid = abs(joint_mass - 1.0) <= 2e-12
    return captures, {
        "valid": valid,
        "joint_probability_mass": float(joint_mass),
        "classes": records,
    }


def build_report(*, arm_a_root: Path, arm_b_root: Path) -> dict[str, Any]:
    arm_a, arm_a_summary = _arm(arm_a_root)
    arm_b, arm_b_summary = _arm(arm_b_root)
    classes = []
    global_group_exact = {group: True for group in GROUP_ORDER}
    for class_one_based in range(1, EXPECTED_CLASSES + 1):
        fields = {}
        class_group_exact = {}
        for group in GROUP_ORDER:
            fields[group] = {
                field: array_metric(arm_a[class_one_based][field], arm_b[class_one_based][field])
                for field in GROUP_FIELDS[group]
            }
            class_group_exact[group] = all(
                record["byte_exact"] for record in fields[group].values()
            )
            global_group_exact[group] &= class_group_exact[group]
        classes.append(
            {
                "class_one_based": class_one_based,
                "group_exact": class_group_exact,
                "first_unequal_group": classify_first_unequal_group(class_group_exact),
                "fields": fields,
            }
        )
    first_unequal = classify_first_unequal_group(global_group_exact)
    gates = {
        "arm_a_valid": arm_a_summary["valid"],
        "arm_b_valid": arm_b_summary["valid"],
        **{f"{group}_exact": exact for group, exact in global_group_exact.items()},
    }
    _require(len(gates) == 9, "fixed gate denominator changed")
    accepted = all(gates.values())
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification": f"first_unequal_group__{first_unequal}",
        "accepted": accepted,
        "stable_cross_engine_operand_attribution_allowed": accepted,
        "first_unequal_group": first_unequal,
        "group_exact": global_group_exact,
        "fixed_metric": {"passing": sum(gates.values()), "evaluated": len(gates), "gates": gates},
        "arms": {"arm_a": arm_a_summary, "arm_b": arm_b_summary},
        "classes": classes,
        "scorecard_change_admissible": False,
        "correlation_used": False,
        "metric_policy": (
            "exact original array bytes grouped in causal order; finite scale-sensitive "
            "deltas are telemetry only; no correlation"
        ),
        "inputs": {
            "arm_a_root": str(arm_a_root.resolve()),
            "arm_b_root": str(arm_b_root.resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a-root", type=Path, required=True)
    parser.add_argument("--arm-b-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    report = build_report(arm_a_root=args.arm_a_root, arm_b_root=args.arm_b_root)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "accepted": report["accepted"],
                "fixed_metric": report["fixed_metric"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
