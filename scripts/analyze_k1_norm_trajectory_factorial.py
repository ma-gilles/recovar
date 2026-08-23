#!/usr/bin/env python3
"""Compare a bounded K=1 normalization factorial at iteration 2.

Each arm must already have the standard K=1 particle-state and FSC audits.
The report aligns iteration-1 normalization factors by immutable source row,
then measures particle-resolved movement toward RELION at iteration 2.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "recovar.em.k1_norm_trajectory_factorial.v1"
GPU_UUID_RE = re.compile(r"GPU UUID\s*:\s*(GPU-[0-9a-f-]+)", re.IGNORECASE)


def _relative_l2(delta: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(np.asarray(reference, dtype=np.float64).reshape(-1)))
    return float(np.linalg.norm(np.asarray(delta, dtype=np.float64).reshape(-1)) / denominator)


def _gpu_uuid(root: Path) -> str | None:
    paths = sorted((root / "provenance").glob("nvidia_smi_*.txt"))
    if len(paths) != 1:
        return None
    match = GPU_UUID_RE.search(paths[0].read_text(errors="replace"))
    return None if match is None else match.group(1)


def _source_aligned_factor(root: Path) -> np.ndarray:
    with np.load(root / "parity" / "iter_001.npz", allow_pickle=False) as payload:
        source_parts = [np.asarray(payload[f"half{half}_original_image_indices"], dtype=np.int64) for half in (1, 2)]
        n_images = sum(part.size for part in source_parts)
        factors = np.full(n_images, np.nan, dtype=np.float32)
        for half, source_indices in zip((1, 2), source_parts, strict=True):
            values = np.asarray(payload[f"half{half}_image_corrections"], dtype=np.float32)
            if values.shape != source_indices.shape:
                raise ValueError(f"half {half} factor/source shape mismatch in {root}")
            factors[source_indices] = values
    if not np.array_equal(np.sort(np.concatenate(source_parts)), np.arange(n_images)):
        raise ValueError(f"source indices are not a complete permutation in {root}")
    if not np.all(np.isfinite(factors)):
        raise ValueError(f"non-finite source-aligned factors in {root}")
    return factors


def _load_arm(root: Path, commit: str | None) -> dict[str, Any]:
    analysis = root / "analysis"
    with np.load(analysis / "K1_PARTICLE_STATE_IT1_IT2_arrays.npz", allow_pickle=False) as payload:
        pmax_recovar = np.asarray(payload["it002_pmax_recovar"], dtype=np.float64)
        pmax_relion = np.asarray(payload["it002_pmax_relion"], dtype=np.float64)
        support_recovar = np.asarray(payload["it002_support_recovar"], dtype=np.int64)
        support_relion = np.asarray(payload["it002_support_relion"], dtype=np.int64)
        pose_error_deg = np.asarray(payload["it002_rotation_geodesic_deg"], dtype=np.float64)
        translation_error = np.asarray(payload["it002_translation_l2"], dtype=np.float64)
        identity_sha256 = str(np.asarray(payload["identity_sha256"]).item())
    fsc = json.loads((analysis / "K1_FSC_IT1_IT2.json").read_text())
    iteration2 = fsc["numbered_iterations"][1]
    factor = _source_aligned_factor(root)
    if factor.shape != pmax_recovar.shape:
        raise ValueError(f"factor/Pmax shape mismatch in {root}")
    return {
        "root": str(root.resolve()),
        "commit": commit,
        "gpu_uuid": _gpu_uuid(root),
        "identity_sha256": identity_sha256,
        "factor": factor,
        "pmax_recovar": pmax_recovar,
        "pmax_relion": pmax_relion,
        "support_recovar": support_recovar,
        "support_relion": support_relion,
        "pose_error_deg": pose_error_deg,
        "translation_error": translation_error,
        "iteration2_merged_signed_fsc_auc": float(iteration2["cross_engine"]["merged"]["signed_fsc_auc"]),
    }


def _arm_metrics(arm: dict[str, Any]) -> dict[str, Any]:
    pmax_delta = arm["pmax_recovar"] - arm["pmax_relion"]
    return {
        "root": arm["root"],
        "commit": arm["commit"],
        "gpu_uuid": arm["gpu_uuid"],
        "identity_sha256": arm["identity_sha256"],
        "iteration2": {
            "pmax_relative_l2": _relative_l2(pmax_delta, arm["pmax_relion"]),
            "pmax_max_abs": float(np.max(np.abs(pmax_delta))),
            "pmax_mean_abs": float(np.mean(np.abs(pmax_delta))),
            "support_mismatch_count": int(np.count_nonzero(arm["support_recovar"] != arm["support_relion"])),
            "pose_error_gt_0p01_deg_count": int(np.count_nonzero(arm["pose_error_deg"] > 0.01)),
            "translation_error_gt_0p01_count": int(np.count_nonzero(arm["translation_error"] > 0.01)),
            "merged_signed_fsc_auc": arm["iteration2_merged_signed_fsc_auc"],
        },
    }


def _movement(baseline: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    if baseline["identity_sha256"] != treatment["identity_sha256"]:
        raise ValueError("particle identity hashes differ between factorial arms")
    if not np.array_equal(baseline["pmax_relion"], treatment["pmax_relion"]):
        raise ValueError("RELION Pmax reference differs between factorial arms")
    if not np.array_equal(baseline["support_relion"], treatment["support_relion"]):
        raise ValueError("RELION support reference differs between factorial arms")

    factor_changed = treatment["factor"] != baseline["factor"]
    baseline_abs = np.abs(baseline["pmax_recovar"] - baseline["pmax_relion"])
    treatment_abs = np.abs(treatment["pmax_recovar"] - treatment["pmax_relion"])
    baseline_support_bad = baseline["support_recovar"] != baseline["support_relion"]
    treatment_support_bad = treatment["support_recovar"] != treatment["support_relion"]

    def count(mask: np.ndarray) -> int:
        return int(np.count_nonzero(mask))

    changed_baseline_mean = float(np.mean(baseline_abs[factor_changed])) if np.any(factor_changed) else None
    changed_treatment_mean = float(np.mean(treatment_abs[factor_changed])) if np.any(factor_changed) else None
    return {
        "same_physical_gpu": baseline["gpu_uuid"] == treatment["gpu_uuid"],
        "factor_changed_count": count(factor_changed),
        "factor_max_abs_change": float(np.max(np.abs(treatment["factor"] - baseline["factor"]))),
        "pmax_abs_error": {
            "improved_count": count(treatment_abs < baseline_abs),
            "exact_equal_count": count(treatment_abs == baseline_abs),
            "worsened_count": count(treatment_abs > baseline_abs),
            "factor_changed_improved_count": count((treatment_abs < baseline_abs) & factor_changed),
            "factor_changed_equal_count": count((treatment_abs == baseline_abs) & factor_changed),
            "factor_changed_worsened_count": count((treatment_abs > baseline_abs) & factor_changed),
            "factor_changed_baseline_mean_abs": changed_baseline_mean,
            "factor_changed_treatment_mean_abs": changed_treatment_mean,
        },
        "support": {
            "baseline_mismatch_count": count(baseline_support_bad),
            "treatment_mismatch_count": count(treatment_support_bad),
            "fixed_count": count(baseline_support_bad & ~treatment_support_bad),
            "new_count": count(~baseline_support_bad & treatment_support_bad),
            "retained_count": count(baseline_support_bad & treatment_support_bad),
        },
        "arm_to_arm_pmax_relative_l2": _relative_l2(
            treatment["pmax_recovar"] - baseline["pmax_recovar"], baseline["pmax_recovar"]
        ),
        "merged_fsc_deficit_ratio_treatment_over_baseline": float(
            (1.0 - treatment["iteration2_merged_signed_fsc_auc"]) / (1.0 - baseline["iteration2_merged_signed_fsc_auc"])
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    rows = []
    for name in ("control", "high_shell", "combined"):
        metrics = report["arms"][name]["iteration2"]
        rows.append(
            f"| {name} | {metrics['pmax_relative_l2']:.12g} | "
            f"{metrics['support_mismatch_count']} | {metrics['pose_error_gt_0p01_deg_count']} | "
            f"{metrics['merged_signed_fsc_auc']:.13f} |"
        )
    return (
        "\n".join(
            [
                "# K=1 normalization two-iteration factorial",
                "",
                "| Arm | Iteration-2 Pmax relative L2 | Support mismatches | Pose >0.01 deg | Merged signed FSC-AUC |",
                "|---|---:|---:|---:|---:|",
                *rows,
                "",
                "This is a bounded localization report, not a fixed-scorecard promotion.",
            ]
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "high-shell", "combined"):
        parser.add_argument(f"--{name}-root", type=Path, required=True)
        parser.add_argument(f"--{name}-commit")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    arms = {
        "control": _load_arm(args.control_root, args.control_commit),
        "high_shell": _load_arm(args.high_shell_root, args.high_shell_commit),
        "combined": _load_arm(args.combined_root, args.combined_commit),
    }
    report = {
        "schema": SCHEMA,
        "scope": "K=1 case 22, two numbered iterations, final pass disabled",
        "acceptance": "descriptive localization only; fixed 34-case scorecard unchanged",
        "arms": {name: _arm_metrics(arm) for name, arm in arms.items()},
        "movement": {
            "control_to_high_shell": _movement(arms["control"], arms["high_shell"]),
            "high_shell_to_combined": _movement(arms["high_shell"], arms["combined"]),
            "control_to_combined": _movement(arms["control"], arms["combined"]),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
