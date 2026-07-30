#!/usr/bin/env python3
"""Gate case-22 serialized-restart map effects with fixed FSC-AUC metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_em_k1_serialized_restart_boundary import (
    CLASSIFICATION as SCORE_BOUNDARY_CLASSIFICATION,
)
from scripts.audit_em_k1_membership_capture_inertness import _sha256
from scripts.summarize_em_completion_bench import (
    _load_recovar_volume,
    _load_relion_volume,
    normalized_fsc_auc,
    shell_fsc,
)

SCHEMA = "em-k1-serialized-restart-map-fsc-v1"
CLASSIFICATION = (
    "serialized_restart_improves_all_case22_iteration2_map_fsc_auc_"
    "without_gt_regression"
)
MAP_LABELS = ("half1", "half2", "merged")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def classify_map_effect(
    *,
    parity_improved: int,
    gt_nondegraded: int,
    expected_maps: int,
    merged_parity_improved: bool,
    merged_gt_nondegraded: bool,
) -> str:
    """Classify fixed signed FSC-AUC deltas without a fitted tolerance."""

    if (
        parity_improved == expected_maps
        and gt_nondegraded == expected_maps
    ):
        return CLASSIFICATION
    if parity_improved == expected_maps:
        return (
            "serialized_restart_improves_all_case22_iteration2_map_"
            "parity_but_regresses_gt_fsc_auc"
        )
    if merged_parity_improved and merged_gt_nondegraded:
        return (
            "serialized_restart_improves_case22_iteration2_merged_"
            "map_parity_only"
        )
    if parity_improved == 0:
        return (
            "serialized_restart_does_not_improve_case22_iteration2_"
            "map_parity"
        )
    return (
        "serialized_restart_has_mixed_case22_iteration2_map_fsc_auc_effect"
    )


def _validate_score_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    _require(
        report.get("schema") == "em-k1-serialized-restart-boundary-v1",
        "serialized-restart score schema changed",
    )
    _require(
        report.get("status") == "complete"
        and report.get("classification_ready") is True,
        "serialized-restart score report is not classification-ready",
    )
    _require(
        report.get("classification") == SCORE_BOUNDARY_CLASSIFICATION,
        "serialized-restart score boundary did not pass its fixed gates",
    )
    fixed = report.get("fixed_metric", {})
    _require(
        fixed.get("evaluated_particles") == 14
        and fixed.get("expected_particles") == 14
        and fixed.get("serialized_restart_dominated") == 14
        and fixed.get("absolute_score_gate_passed") == 14,
        "serialized-restart score fixed metric did not pass 14/14",
    )
    return report


def _paths(
    *,
    recovar_root: Path,
    fresh_relion_root: Path,
    restart_relion_root: Path,
    relion_iteration: int,
) -> dict[str, dict[str, Path]]:
    return {
        "recovar": {
            "half1": recovar_root / "recovar" / "final_half1.mrc",
            "half2": recovar_root / "recovar" / "final_half2.mrc",
            "merged": recovar_root / "recovar" / "final_merged.mrc",
        },
        "fresh_relion": {
            label: (
                fresh_relion_root
                / "relion"
                / f"run_it{relion_iteration:03d}_half{half}_class001.mrc"
            )
            for label, half in (("half1", 1), ("half2", 2))
        },
        "restart_relion": {
            label: (
                restart_relion_root
                / "relion"
                / f"run_it{relion_iteration:03d}_half{half}_class001.mrc"
            )
            for label, half in (("half1", 1), ("half2", 2))
        },
    }


def _load_maps(
    paths: dict[str, dict[str, Path]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    loaded: dict[str, dict[str, np.ndarray]] = {}
    hashes: dict[str, str] = {}
    for arm, arm_paths in paths.items():
        loaded[arm] = {}
        loader = (
            _load_recovar_volume if arm == "recovar" else _load_relion_volume
        )
        for label, path in arm_paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            volume = np.asarray(loader(path), dtype=np.float64)
            _require(
                volume.ndim == 3 and len(set(volume.shape)) == 1,
                f"expected cubic 3-D map at {path}, got {volume.shape}",
            )
            _require(
                np.all(np.isfinite(volume)),
                f"map contains non-finite values: {path}",
            )
            loaded[arm][label] = volume
            hashes[str(path.resolve())] = _sha256(path)

    for arm in ("fresh_relion", "restart_relion"):
        loaded[arm]["merged"] = 0.5 * (
            loaded[arm]["half1"] + loaded[arm]["half2"]
        )
    expected_shape = loaded["recovar"]["merged"].shape
    for arm, maps in loaded.items():
        for label, volume in maps.items():
            _require(
                volume.shape == expected_shape,
                f"map shape mismatch for {arm} {label}: "
                f"{volume.shape} != {expected_shape}",
            )
    return loaded, hashes


def _metric(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    curve = np.asarray(shell_fsc(lhs, rhs), dtype=np.float64).reshape(-1)
    _require(curve.size > 1, "map FSC produced too few shells")
    finite_non_dc = np.isfinite(curve[1:])
    _require(
        bool(np.all(finite_non_dc)),
        "map FSC has a non-finite non-DC shell",
    )
    auc = float(normalized_fsc_auc(curve))
    _require(math.isfinite(auc), "map FSC-AUC is non-finite")
    return {
        "fsc_auc_non_dc": auc,
        "n_shells": int(curve.size),
        "fsc": [float(value) if math.isfinite(value) else None for value in curve],
    }


def build_report(
    *,
    score_analysis_json: Path,
    recovar_root: Path,
    fresh_relion_root: Path,
    restart_relion_root: Path,
    gt_volume: Path,
    relion_iteration: int,
) -> dict[str, Any]:
    """Build the predeclared map-level causal gate for the restart arm."""

    _require(relion_iteration >= 1, "RELION iteration must be positive")
    score_report = _validate_score_report(score_analysis_json)
    paths = _paths(
        recovar_root=recovar_root,
        fresh_relion_root=fresh_relion_root,
        restart_relion_root=restart_relion_root,
        relion_iteration=relion_iteration,
    )
    loaded, hashes = _load_maps(paths)
    if not gt_volume.is_file():
        raise FileNotFoundError(gt_volume)
    gt = np.asarray(_load_recovar_volume(gt_volume), dtype=np.float64)
    _require(
        gt.shape == loaded["recovar"]["merged"].shape,
        f"GT/map shape mismatch: {gt.shape} != "
        f"{loaded['recovar']['merged'].shape}",
    )
    _require(np.all(np.isfinite(gt)), f"GT contains non-finite values: {gt_volume}")
    hashes[str(gt_volume.resolve())] = _sha256(gt_volume)

    comparisons: dict[str, Any] = {}
    parity_improved = 0
    gt_nondegraded = 0
    for label in MAP_LABELS:
        recovar_fresh = _metric(
            loaded["recovar"][label],
            loaded["fresh_relion"][label],
        )
        recovar_restart = _metric(
            loaded["recovar"][label],
            loaded["restart_relion"][label],
        )
        fresh_restart = _metric(
            loaded["fresh_relion"][label],
            loaded["restart_relion"][label],
        )
        gt_recovar = _metric(gt, loaded["recovar"][label])
        gt_fresh = _metric(gt, loaded["fresh_relion"][label])
        gt_restart = _metric(gt, loaded["restart_relion"][label])
        parity_delta = float(
            recovar_restart["fsc_auc_non_dc"]
            - recovar_fresh["fsc_auc_non_dc"]
        )
        gt_delta = float(
            gt_restart["fsc_auc_non_dc"] - gt_fresh["fsc_auc_non_dc"]
        )
        current_parity_improved = parity_delta > 0.0
        current_gt_nondegraded = gt_delta >= 0.0
        parity_improved += int(current_parity_improved)
        gt_nondegraded += int(current_gt_nondegraded)
        comparisons[label] = {
            "recovar_vs_fresh_relion": recovar_fresh,
            "recovar_vs_restart_relion": recovar_restart,
            "fresh_vs_restart_relion": fresh_restart,
            "gt_vs_recovar": gt_recovar,
            "gt_vs_fresh_relion": gt_fresh,
            "gt_vs_restart_relion": gt_restart,
            "restart_minus_fresh_parity_fsc_auc": parity_delta,
            "restart_minus_fresh_gt_fsc_auc": gt_delta,
            "parity_strictly_improved": current_parity_improved,
            "gt_nondegraded": current_gt_nondegraded,
        }

    merged = comparisons["merged"]
    classification = classify_map_effect(
        parity_improved=parity_improved,
        gt_nondegraded=gt_nondegraded,
        expected_maps=len(MAP_LABELS),
        merged_parity_improved=merged["parity_strictly_improved"],
        merged_gt_nondegraded=merged["gt_nondegraded"],
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification_ready": True,
        "classification": classification,
        "metric_policy": (
            "signed shellwise FSC and normalized non-DC FSC-AUC only; "
            "fixed half1/half2/merged denominator; parity improvement is "
            "restart-minus-fresh > 0; GT non-degradation is "
            "restart-minus-fresh >= 0; no fitted tolerance; no correlation"
        ),
        "relion_iteration": relion_iteration,
        "fixed_metric": {
            "parity_strictly_improved": parity_improved,
            "gt_nondegraded": gt_nondegraded,
            "evaluated_maps": len(MAP_LABELS),
            "expected_maps": 3,
        },
        "score_boundary": {
            "path": str(score_analysis_json.resolve()),
            "sha256": _sha256(score_analysis_json),
            "classification": score_report["classification"],
            "fixed_metric": score_report["fixed_metric"],
        },
        "roots": {
            "recovar": str(recovar_root.resolve()),
            "fresh_relion": str(fresh_relion_root.resolve()),
            "restart_relion": str(restart_relion_root.resolve()),
        },
        "gt_volume": str(gt_volume.resolve()),
        "comparisons": comparisons,
        "artifact_sha256": hashes,
        "notes": [
            (
                "RELION maps were converted to RECOVAR coordinates through "
                "recovar.utils.helpers.load_relion_volume."
            ),
            (
                "RECOVAR and GT maps were loaded in their native RECOVAR "
                "coordinate frame."
            ),
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-analysis-json", type=Path, required=True)
    parser.add_argument("--recovar-root", type=Path, required=True)
    parser.add_argument("--fresh-relion-root", type=Path, required=True)
    parser.add_argument("--restart-relion-root", type=Path, required=True)
    parser.add_argument("--gt-volume", type=Path, required=True)
    parser.add_argument("--relion-iteration", type=int, default=2)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    _require(
        not args.output_json.exists(),
        f"refusing to overwrite report: {args.output_json}",
    )
    report = build_report(
        score_analysis_json=args.score_analysis_json,
        recovar_root=args.recovar_root,
        fresh_relion_root=args.fresh_relion_root,
        restart_relion_root=args.restart_relion_root,
        gt_volume=args.gt_volume,
        relion_iteration=args.relion_iteration,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report["fixed_metric"], indent=2, sort_keys=True))
    print(report["classification"])


if __name__ == "__main__":
    main()
