#!/usr/bin/env python3
"""Gate a passive RELION capture on half-map and merged-map FSC-AUC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.audit_em_k1_membership_capture_inertness import (
    DEFAULT_FSC_AUC_THRESHOLD,
    _load_map,
    _metrics,
    _sha256,
)

SCHEMA = "em-relion-capture-inertness-v1"


def _path(root: Path, iteration: int, half: int) -> Path:
    return root / "relion" / f"run_it{iteration:03d}_half{half}_class001.mrc"


def build_report(
    *,
    control_root: Path,
    capture_root: Path,
    relion_iteration: int,
    fsc_auc_threshold: float,
) -> dict[str, Any]:
    if relion_iteration < 1:
        raise ValueError("RELION iteration must be positive")
    if not 0.0 < fsc_auc_threshold <= 1.0:
        raise ValueError("FSC-AUC threshold must be in (0, 1]")
    loaded: dict[str, list[Any]] = {"control": [], "capture": []}
    comparisons: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for half in (1, 2):
        paths = {
            "control": _path(control_root, relion_iteration, half),
            "capture": _path(capture_root, relion_iteration, half),
        }
        volumes = {arm: _load_map(path) for arm, path in paths.items()}
        for arm, path in paths.items():
            loaded[arm].append(volumes[arm])
            hashes[str(path.resolve())] = _sha256(path)
        row = _metrics(volumes["control"], volumes["capture"])
        row["passed"] = row["fsc_auc_non_dc"] >= fsc_auc_threshold
        comparisons[f"half{half}"] = row
    merged = _metrics(
        0.5 * (loaded["control"][0] + loaded["control"][1]),
        0.5 * (loaded["capture"][0] + loaded["capture"][1]),
    )
    merged["passed"] = merged["fsc_auc_non_dc"] >= fsc_auc_threshold
    comparisons["merged"] = merged
    passed = all(row["passed"] for row in comparisons.values())
    return {
        "schema": SCHEMA,
        "status": "pass" if passed else "rejected",
        "metric_policy": (
            "shellwise FSC/FSC-AUC only for acceptance; "
            "relative L2 and max-absolute error are secondary; no correlation"
        ),
        "control_root": str(control_root.resolve()),
        "capture_root": str(capture_root.resolve()),
        "relion_iteration": relion_iteration,
        "fsc_auc_non_dc_threshold": fsc_auc_threshold,
        "comparisons": comparisons,
        "strict_gate": {
            "passed": sum(row["passed"] for row in comparisons.values()),
            "evaluated": len(comparisons),
            "expected": 3,
        },
        "capture_inertness_qualified": passed,
        "artifact_sha256": hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--relion-iteration", type=int, default=2)
    parser.add_argument(
        "--fsc-auc-threshold",
        type=float,
        default=DEFAULT_FSC_AUC_THRESHOLD,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")
    report = build_report(
        control_root=args.control_root,
        capture_root=args.capture_root,
        relion_iteration=args.relion_iteration,
        fsc_auc_threshold=args.fsc_auc_threshold,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
