#!/usr/bin/env python3
"""Audit RELION joint class-direction priors against RECOVAR's split form."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import starfile

SCHEMA = "recovar-k4-joint-direction-prior-audit-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    _require(left.shape == right.shape, "metric shapes differ")
    finite = np.isfinite(left) & np.isfinite(right)
    same_nonfinite = np.array_equal(
        np.isneginf(left),
        np.isneginf(right),
    )
    mismatch = left.view(np.uint32) != right.view(np.uint32)
    mismatch_flat_indices = np.flatnonzero(mismatch.reshape(-1))
    delta = np.zeros(left.shape, dtype=np.float64)
    delta[finite] = right[finite].astype(np.float64) - left[finite].astype(
        np.float64
    )
    return {
        "bitwise_exact": bool(not np.any(mismatch)),
        "same_negative_infinity_mask": bool(same_nonfinite),
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "mismatch_flat_indices": mismatch_flat_indices.astype(
            np.int64
        ).tolist(),
        "first_mismatch_flat_index": (
            int(mismatch_flat_indices[0])
            if mismatch_flat_indices.size
            else None
        ),
        "total_count": int(left.size),
        "maximum_absolute_delta": float(
            np.max(np.abs(delta[finite])) if np.any(finite) else 0.0
        ),
    }


def audit_joint_direction_prior(
    direction_prior: np.ndarray,
) -> dict[str, Any]:
    """Compare direct joint logs with the current class/conditional split."""

    joint = np.asarray(direction_prior, dtype=np.float32)
    _require(
        joint.ndim == 2 and joint.shape[0] > 1 and joint.shape[1] > 0,
        "direction prior must have shape (n_classes, n_directions)",
    )
    _require(
        np.all(np.isfinite(joint)) and np.all(joint >= 0.0),
        "direction-prior entries must be finite and nonnegative",
    )
    row_sums = joint.sum(axis=1, dtype=np.float64)
    _require(np.all(row_sums > 0.0), "each class row must have positive mass")
    total = float(row_sums.sum())
    _require(np.isfinite(total) and total > 0.0, "total prior mass is invalid")

    conditional = (
        joint / row_sums[:, None].astype(np.float32)
    ).astype(np.float32)
    class_weights = (row_sums / total).astype(np.float32)

    conditional_log = np.full(joint.shape, -np.inf, dtype=np.float32)
    split_log = np.full(joint.shape, -np.inf, dtype=np.float32)
    direct_log = np.full(joint.shape, -np.inf, dtype=np.float32)
    normalized_direct_log = np.full(joint.shape, -np.inf, dtype=np.float32)
    positive = joint > 0.0
    conditional_log[positive] = np.log(conditional[positive]).astype(
        np.float32
    )
    class_log = np.log(class_weights).astype(np.float32)
    class_log_expanded = np.broadcast_to(class_log[:, None], joint.shape)
    split_log[positive] = (
        conditional_log[positive] + class_log_expanded[positive]
    ).astype(np.float32)
    direct_log[positive] = np.log(joint[positive]).astype(np.float32)
    normalized_joint = (joint / np.float32(total)).astype(np.float32)
    normalized_direct_log[positive] = np.log(
        normalized_joint[positive]
    ).astype(np.float32)

    split_vs_direct = _metric(direct_log, split_log)
    split_vs_normalized_direct = _metric(
        normalized_direct_log,
        split_log,
    )
    class_metrics = []
    for class_index in range(joint.shape[0]):
        class_metrics.append(
            {
                "class_one_based": class_index + 1,
                "joint_mass_float64": float(row_sums[class_index]),
                "class_weight_float32": float(class_weights[class_index]),
                "split_vs_direct": _metric(
                    direct_log[class_index],
                    split_log[class_index],
                ),
                "split_vs_normalized_direct": _metric(
                    normalized_direct_log[class_index],
                    split_log[class_index],
                ),
            }
        )

    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification": (
            "exact_float32_joint_direction_log_split"
            if split_vs_direct["bitwise_exact"]
            else "float32_joint_direction_log_split_mismatch"
        ),
        "n_classes": int(joint.shape[0]),
        "n_directions": int(joint.shape[1]),
        "total_joint_mass_float64": total,
        "row_sums_float64": row_sums.tolist(),
        "class_weights_float32": class_weights.tolist(),
        "split_vs_relion_direct_joint_log": split_vs_direct,
        "split_vs_analytically_normalized_direct_joint_log": (
            split_vs_normalized_direct
        ),
        "per_class": class_metrics,
        "causal_claim_admissible": False,
        "scorecard_change_admissible": False,
        "interpretation": (
            "RELION's accelerated NOPRIOR path logs the original joint "
            "pdf_direction entry once. RECOVAR currently logs a row-normalized "
            "conditional and class mass separately, then adds them. This audit "
            "measures the resulting float32 operand difference but does not "
            "establish a posterior, BPref, reduction, or map effect."
        ),
    }


def _read_model_direction_prior(model_path: Path) -> np.ndarray:
    model = starfile.read(model_path)
    pattern = re.compile(r"model_pdf_orient_class_(\d+)")
    indexed = []
    for key, table in model.items():
        match = pattern.fullmatch(str(key))
        if match is None:
            continue
        indexed.append(
            (
                int(match.group(1)),
                np.asarray(
                    table["rlnOrientationDistribution"],
                    dtype=np.float32,
                ),
            )
        )
    _require(bool(indexed), "model has no class direction-prior tables")
    indexed.sort(key=lambda item: item[0])
    expected = list(range(1, len(indexed) + 1))
    _require(
        [index for index, _ in indexed] == expected,
        "class direction-prior tables are not contiguous",
    )
    return np.stack([values for _, values in indexed], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_joint_direction_prior(
        _read_model_direction_prior(args.model)
    )
    report["input"] = {
        "path": str(args.model.resolve()),
        "sha256": _sha256(args.model),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
