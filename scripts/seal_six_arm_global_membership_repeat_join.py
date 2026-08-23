#!/usr/bin/env python3
"""Seal the exact six-arm case11 membership/repeat reconciliation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from recovar.em.global_winner_analysis import read_relion_identity_classes
from recovar.em.global_winner_summary import sha256_file

ARM_NAMES = (
    "relion_controlA",
    "relion_controlB",
    "relion_capture",
    "recovar_controlA",
    "recovar_controlB",
    "recovar_capture",
)


def _load_arm(root: Path, arm: str) -> tuple[np.ndarray, Path]:
    if arm.startswith("relion_"):
        path = root / arm / "run_it001_data.star"
        identity_rows, _classes = read_relion_identity_classes(path)
        rows = sorted(identity_rows.values(), key=lambda item: item[1])
        identity = np.asarray([row[1] for row in rows], dtype=np.int64)
        classes = np.asarray([row[2] for row in rows], dtype=np.int32)
        if not np.array_equal(identity, np.arange(identity.size, dtype=np.int64)):
            raise ValueError(f"{arm} does not contain the exact original-stack identity set")
        return classes, path
    path = root / arm / "refinement_results.npz"
    with np.load(path, allow_pickle=False) as payload:
        classes = np.asarray(payload["class_assignments_by_image_iter_000"], dtype=np.int32)
    return classes, path


def _score_geometry_summary(root: Path) -> dict:
    analysis_path = root / "analyze_six_arm.py"
    spec = importlib.util.spec_from_file_location("case11_six_arm_analysis", analysis_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {analysis_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.score_geometry_report(root)
    coarse = report["coarse_scores"]
    geometry = report["geometry_prerequisites"]
    return {
        "status": report["status"],
        "coarse_candidate_count": geometry["coarse_candidate_count"],
        "coarse_candidate_bijection": geometry["coarse_candidate_bijection"],
        "coarse_rotation_max_abs": geometry["coarse_rotation_physical_gate"]["max_abs"],
        "coarse_translation_max_abs_px": geometry["coarse_translation_physical_gate"]["max_abs"],
        "recovar_margin": coarse["recovar_margin"],
        "relion_margin": coarse["relion_margin"],
        "centered_score_residual_envelope": coarse["centered_score_residual_envelope"],
        "centered_score_relative_l1": coarse["centered_sign_converted_score_metrics"]["rel_l1"],
        "classification": coarse["classification"],
        "classification_caveat": (
            "strong f32 evidence only; not a final numerical classification without native-repeat "
            "and recomputed float64/complex128 score controls"
        ),
        "hashes": report["hashes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    arrays = {}
    sources = {}
    for arm in ARM_NAMES:
        arrays[arm], source = _load_arm(root, arm)
        sources[arm] = {
            "path": str(source),
            "sha256": sha256_file(source),
            "counts": np.bincount(arrays[arm], minlength=4).tolist(),
            "winner_sha256": hashlib.sha256(arrays[arm].tobytes()).hexdigest(),
        }
    identity_count = arrays[ARM_NAMES[0]].size
    if identity_count != 10_000 or any(array.shape != (identity_count,) for array in arrays.values()):
        raise ValueError("six-arm seal requires exactly 10,000 identities in every arm")

    pairwise = []
    for left_index, left in enumerate(ARM_NAMES):
        for right in ARM_NAMES[left_index + 1 :]:
            differing = np.flatnonzero(arrays[left] != arrays[right])
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "mismatch_count": int(differing.size),
                    "mismatches": [
                        {
                            "original_stack_zero_based_index": int(index),
                            "original_stack_one_based_index": int(index + 1),
                            "image_name": f"{index + 1}@particles.128.mrcs",
                            "left_class_zero_based": int(arrays[left][index]),
                            "right_class_zero_based": int(arrays[right][index]),
                        }
                        for index in differing
                    ],
                }
            )

    relion_exact = all(
        np.array_equal(arrays["relion_controlA"], arrays[arm]) for arm in ("relion_controlB", "relion_capture")
    )
    p6325 = 6325
    p7915 = 7915
    result = {
        "schema": "six_arm_global_membership_repeat_join_v1",
        "identity": {
            "convention": "original stack zero-based; image name is one-based stack index",
            "count": identity_count,
            "sha256": hashlib.sha256(np.arange(identity_count, dtype=np.int64).tobytes()).hexdigest(),
        },
        "class_index_convention": "zero_based in arrays/report; add one for RELION STAR presentation",
        "arms": sources,
        "pairwise": pairwise,
        "classification": {
            "relion_repeat_exact": relion_exact,
            "recovar_repeat_variable_identity": {
                "original_stack_zero_based_index": p6325,
                "image_name": "6326@particles.128.mrcs",
                "controlA_class_zero_based": int(arrays["recovar_controlA"][p6325]),
                "controlB_class_zero_based": int(arrays["recovar_controlB"][p6325]),
                "capture_class_zero_based": int(arrays["recovar_capture"][p6325]),
                "interpretation": "within observed RECOVAR native-repeat envelope",
            },
            "recurrent_cross_engine_identity": {
                "original_stack_zero_based_index": p7915,
                "image_name": "7916@particles.128.mrcs",
                "relion_class_zero_based": int(arrays["relion_controlA"][p7915]),
                "recovar_classes_zero_based": {
                    arm: int(arrays[arm][p7915]) for arm in ("recovar_controlA", "recovar_controlB", "recovar_capture")
                },
                "interpretation": "recurrent cross-engine mismatch in all observed repeats",
            },
            "sole_mismatch_claim_policy": (
                "A one-particle cross-engine claim is valid only for RECOVAR controlB; "
                "controlA/capture also differ at repeat-sensitive particle 6326."
            ),
        },
        "p7916_score_geometry": _score_geometry_summary(root),
        "metric_policy": "exact membership/array metrics; score geometry uses exact arrays; no correlation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "relion_repeat_exact": relion_exact}))


if __name__ == "__main__":
    main()
