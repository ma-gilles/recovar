#!/usr/bin/env python3
"""Compare K-class joint support with RELION-style float32 raw weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_f32_fine_reconstruction_probs,
)

SCHEMA = "recovar.em_k4_joint_f32_support_probe.v1"
EXPECTED_CLASSES = 4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _float32_bits(value: jax.Array) -> int:
    scalar = np.asarray(value, dtype=np.float32)
    _require(scalar.shape == (), "expected a scalar float32 value")
    return int(scalar.view(np.uint32))


def _ordered_float32_int(value: np.float32) -> int:
    """Map a finite float32 to an integer ordered by numeric value."""

    bits = int(np.asarray(value, dtype=np.float32).view(np.uint32))
    if bits & 0x80000000:
        return 0xFFFFFFFF - bits
    return bits + 0x80000000


def _load_captures(capture_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = sorted(capture_root.glob("pass2_orig*_class*_cs*.npz"))
    _require(len(paths) == EXPECTED_CLASSES, "capture root must contain four class files")

    captures: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    identity: tuple[int, int] | None = None
    for class_one_based, path in enumerate(paths, 1):
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "original_index",
                "class_index",
                "current_size",
                "scores_with_prior",
                "candidate_mask",
                "reconstruction_mask",
                "reconstruction_n_significant",
            }
            _require(required.issubset(archive.files), f"{path.name} is missing required fields")
            original_index = int(np.asarray(archive["original_index"]))
            current_size = int(np.asarray(archive["current_size"]))
            class_index = int(np.asarray(archive["class_index"]))
            _require(class_index == class_one_based - 1, "class ordering changed")
            if identity is None:
                identity = (original_index, current_size)
            _require(identity == (original_index, current_size), "capture identity/current size changed")

            stored_scores = np.asarray(archive["scores_with_prior"])
            scores = stored_scores.astype(np.float32)
            candidate = np.asarray(archive["candidate_mask"], dtype=bool)
            current = np.asarray(archive["reconstruction_mask"], dtype=bool)
            _require(scores.shape == candidate.shape == current.shape, "capture shapes disagree")
            finite = np.isfinite(stored_scores)
            _require(np.all(finite[candidate]), "active candidate has a nonfinite score")
            _require(
                np.array_equal(stored_scores[finite], scores.astype(np.float64)[finite]),
                "scores_with_prior is not an exact float32 capture",
            )
            _require(not np.any(current & ~candidate), "current support contains an inactive candidate")
            _require(
                int(np.asarray(archive["reconstruction_n_significant"]))
                == int(np.count_nonzero(current)),
                "stored significant count disagrees with the support mask",
            )
            captures.append(
                {
                    "scores": np.where(candidate, scores, -np.inf).reshape(-1),
                    "candidate": candidate.reshape(-1),
                    "current": current.reshape(-1),
                    "shape": candidate.shape,
                }
            )
            input_records.append(
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "class_one_based": class_one_based,
                    "active_candidates": int(np.count_nonzero(candidate)),
                    "current_significant": int(np.count_nonzero(current)),
                }
            )
    return captures, input_records


def build_report(
    *,
    capture_root: Path,
    adaptive_fraction: float,
    repetitions: int,
    source_commit: str | None = None,
    gpu_uuid: str | None = None,
) -> dict[str, Any]:
    """Build a non-scoring support counterfactual from four class captures."""

    _require(0.0 < adaptive_fraction < 1.0, "adaptive_fraction must be in (0, 1)")
    _require(repetitions > 0, "repetitions must be positive")
    captures, input_records = _load_captures(capture_root)
    joint_scores = np.concatenate([capture["scores"] for capture in captures])[None, :]
    joint_candidate = np.concatenate([capture["candidate"] for capture in captures])
    joint_current = np.concatenate([capture["current"] for capture in captures])

    scores_device = jnp.asarray(joint_scores, dtype=jnp.float32)
    runs: list[dict[str, Any]] = []
    for _ in range(repetitions):
        _probs, mask, n_significant, sum_weight, threshold = (
            _relion_f32_fine_reconstruction_probs(
                scores_device,
                adaptive_fraction=float(adaptive_fraction),
            )
        )
        mask_np = np.asarray(mask[0], dtype=bool)
        runs.append(
            {
                "mask": mask_np,
                "n_significant": int(np.asarray(n_significant[0])),
                "sum_weight_bits": _float32_bits(sum_weight[0]),
                "threshold_bits": _float32_bits(threshold[0]),
                "mask_sha256": hashlib.sha256(mask_np.tobytes()).hexdigest(),
            }
        )

    reference = runs[0]
    _require(
        all(np.array_equal(reference["mask"], run["mask"]) for run in runs[1:]),
        "float32 support mask did not repeat exactly",
    )
    _require(
        all(reference["sum_weight_bits"] == run["sum_weight_bits"] for run in runs[1:]),
        "float32 sum_weight did not repeat exactly",
    )
    _require(
        all(reference["threshold_bits"] == run["threshold_bits"] for run in runs[1:]),
        "float32 threshold did not repeat exactly",
    )
    counterfactual = reference["mask"]
    _require(not np.any(counterfactual & ~joint_candidate), "counterfactual retained an inactive candidate")
    _require(reference["n_significant"] == int(np.count_nonzero(counterfactual)), "counterfactual count changed")

    scores_flat = joint_scores[0]
    retained_scores = scores_flat[counterfactual]
    excluded_active_scores = scores_flat[joint_candidate & ~counterfactual]
    _require(retained_scores.size > 0, "counterfactual support is empty")
    min_retained_score = np.float32(np.min(retained_scores))
    if excluded_active_scores.size:
        max_excluded_score = np.float32(np.max(excluded_active_scores))
        score_margin = float(np.float64(min_retained_score) - np.float64(max_excluded_score))
        score_margin_ulps = _ordered_float32_int(min_retained_score) - _ordered_float32_int(
            max_excluded_score,
        )
        _require(score_margin >= 0.0 and score_margin_ulps >= 0, "support score boundary is inverted")
        max_excluded_score_out: float | None = float(max_excluded_score)
        max_excluded_score_bits: int | None = int(max_excluded_score.view(np.uint32))
        score_margin_ulps_out: int | None = int(score_margin_ulps)
        excluded_ties_at_max: int | None = int(np.count_nonzero(excluded_active_scores == max_excluded_score))
    else:
        max_excluded_score_out = None
        max_excluded_score_bits = None
        score_margin = None
        score_margin_ulps_out = None
        excluded_ties_at_max = None

    offsets = np.cumsum([0] + [capture["scores"].size for capture in captures])
    class_records = []
    for class_index, capture in enumerate(captures):
        class_slice = slice(offsets[class_index], offsets[class_index + 1])
        current = joint_current[class_slice]
        proposed = counterfactual[class_slice]
        mismatch = current != proposed
        class_records.append(
            {
                "class_one_based": class_index + 1,
                "shape": list(capture["shape"]),
                "current_significant": int(np.count_nonzero(current)),
                "f32_raw_weight_significant": int(np.count_nonzero(proposed)),
                "mask_mismatch_count": int(np.count_nonzero(mismatch)),
                "current_only_count": int(np.count_nonzero(current & ~proposed)),
                "f32_only_count": int(np.count_nonzero(proposed & ~current)),
                "first_mismatch_flat_index": (
                    int(np.flatnonzero(mismatch)[0]) if np.any(mismatch) else None
                ),
            }
        )

    exact = np.array_equal(counterfactual, joint_current)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete",
        "classification": (
            "joint_f32_raw_weight_support_matches_current_probability_support"
            if exact
            else "joint_f32_raw_weight_support_differs_from_current_probability_support"
        ),
        "adaptive_fraction": float(adaptive_fraction),
        "device": str(jax.devices()[0]),
        "repetitions": repetitions,
        "repeat_mask_exact": True,
        "current_total_significant": int(np.count_nonzero(joint_current)),
        "f32_raw_weight_total_significant": int(np.count_nonzero(counterfactual)),
        "joint_mask_mismatch_count": int(np.count_nonzero(counterfactual != joint_current)),
        "current_only_count": int(np.count_nonzero(joint_current & ~counterfactual)),
        "f32_only_count": int(np.count_nonzero(counterfactual & ~joint_current)),
        "f32_sum_weight_bits": reference["sum_weight_bits"],
        "f32_threshold_bits": reference["threshold_bits"],
        "f32_mask_sha256": reference["mask_sha256"],
        "support_score_boundary": {
            "min_retained_score_f32": float(min_retained_score),
            "min_retained_score_bits": int(min_retained_score.view(np.uint32)),
            "retained_ties_at_min": int(np.count_nonzero(retained_scores == min_retained_score)),
            "max_excluded_active_score_f32": max_excluded_score_out,
            "max_excluded_active_score_bits": max_excluded_score_bits,
            "excluded_ties_at_max": excluded_ties_at_max,
            "retained_minus_excluded_score_margin": score_margin,
            "score_margin_float32_ulps": score_margin_ulps_out,
        },
        "classes": class_records,
        "inputs": input_records,
        "interpretation": (
            "A RECOVAR capture-only counterfactual. It tests whether the existing GPU "
            "float32 raw-weight helper changes support relative to current joint "
            "probability thresholding. It is not a native RELION comparison, does not "
            "establish trajectory causality, and cannot authorize production."
        ),
        "scorecard_change_admissible": False,
        "production_authorized": False,
        "correlation_used": False,
    }
    if source_commit is not None:
        report["source_commit"] = source_commit
    if gpu_uuid is not None:
        report["gpu_uuid"] = gpu_uuid
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--adaptive-fraction", default=0.999, type=float)
    parser.add_argument("--repetitions", default=4, type=int)
    parser.add_argument("--source-commit")
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = build_report(
        capture_root=args.capture_root,
        adaptive_fraction=args.adaptive_fraction,
        repetitions=args.repetitions,
        source_commit=args.source_commit,
        gpu_uuid=args.gpu_uuid,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
