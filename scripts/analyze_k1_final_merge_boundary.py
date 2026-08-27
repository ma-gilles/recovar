#!/usr/bin/env python3
"""Localize a K=1 final FSC failure at the explicit merged-map boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.audit_k1_fsc_trajectory import (
        _load_recovar_volume,
        _load_relion_volume,
        _map_metric,
    )
except ModuleNotFoundError:  # Support direct execution from the repository root.
    from audit_k1_fsc_trajectory import (
        _load_recovar_volume,
        _load_relion_volume,
        _map_metric,
    )


SCHEMA = "recovar.em.k1_final_merge_boundary.v1"
FIXED_FSC_AUC_MIN = 0.995


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_report_from_arrays(
    *,
    recovar_half1: np.ndarray,
    recovar_half2: np.ndarray,
    recovar_merged: np.ndarray,
    relion_half1: np.ndarray,
    relion_half2: np.ndarray,
    relion_merged: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays = {
        "recovar_half1": np.asarray(recovar_half1),
        "recovar_half2": np.asarray(recovar_half2),
        "recovar_merged": np.asarray(recovar_merged),
        "relion_half1": np.asarray(relion_half1),
        "relion_half2": np.asarray(relion_half2),
        "relion_merged": np.asarray(relion_merged),
    }
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"all final maps must have the same shape, found {sorted(shapes)}")

    recovar_half_average = 0.5 * (arrays["recovar_half1"] + arrays["recovar_half2"])
    relion_half_average = 0.5 * (arrays["relion_half1"] + arrays["relion_half2"])
    shellwise: dict[str, np.ndarray] = {}

    def metric(label: str, lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
        return _map_metric(
            lhs,
            rhs,
            sign_invariant=False,
            shellwise_key=label,
            shellwise=shellwise,
        )

    comparisons = {
        "cross_half1": metric(
            "cross_half1", arrays["recovar_half1"], arrays["relion_half1"]
        ),
        "cross_half2": metric(
            "cross_half2", arrays["recovar_half2"], arrays["relion_half2"]
        ),
        "cross_half_average": metric(
            "cross_half_average", recovar_half_average, relion_half_average
        ),
        "cross_explicit_merged": metric(
            "cross_explicit_merged", arrays["recovar_merged"], arrays["relion_merged"]
        ),
        "recovar_explicit_vs_half_average": metric(
            "recovar_explicit_vs_half_average",
            arrays["recovar_merged"],
            recovar_half_average,
        ),
        "relion_explicit_vs_half_average": metric(
            "relion_explicit_vs_half_average",
            arrays["relion_merged"],
            relion_half_average,
        ),
        "recovar_explicit_vs_relion_half_average": metric(
            "recovar_explicit_vs_relion_half_average",
            arrays["recovar_merged"],
            relion_half_average,
        ),
        "recovar_half_average_vs_relion_explicit": metric(
            "recovar_half_average_vs_relion_explicit",
            recovar_half_average,
            arrays["relion_merged"],
        ),
    }
    half_average_passes = (
        float(comparisons["cross_half_average"]["fsc_auc"]) >= FIXED_FSC_AUC_MIN
    )
    explicit_merged_passes = (
        float(comparisons["cross_explicit_merged"]["fsc_auc"]) >= FIXED_FSC_AUC_MIN
    )
    if half_average_passes and not explicit_merged_passes:
        classification = "failure_appears_only_at_explicit_merged_product_comparison"
    elif not half_average_passes:
        classification = "half_map_errors_already_fail_when_averaged"
    else:
        classification = "no_fixed_merged_fsc_gate_failure"

    return (
        {
            "schema": SCHEMA,
            "status": "complete",
            "classification": classification,
            "metric_policy": (
                "signed shellwise FSC and normalized non-DC FSC-AUC; no correlation; "
                "diagnostic only and not a scorecard promotion"
            ),
            "fixed_fsc_auc_min": FIXED_FSC_AUC_MIN,
            "shape": list(next(iter(shapes))),
            "half_average_cross_engine_passes": half_average_passes,
            "explicit_merged_cross_engine_passes": explicit_merged_passes,
            "comparisons": comparisons,
        },
        shellwise,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "recovar_half1",
        "recovar_half2",
        "recovar_merged",
        "relion_half1",
        "relion_half2",
        "relion_merged",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-shellwise-npz", required=True, type=Path)
    args = parser.parse_args()

    paths = {
        "recovar_half1": args.recovar_half1.resolve(),
        "recovar_half2": args.recovar_half2.resolve(),
        "recovar_merged": args.recovar_merged.resolve(),
        "relion_half1": args.relion_half1.resolve(),
        "relion_half2": args.relion_half2.resolve(),
        "relion_merged": args.relion_merged.resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    if args.output_json.exists() or args.output_shellwise_npz.exists():
        raise FileExistsError("refusing to overwrite merge-boundary output")

    report, shellwise = build_report_from_arrays(
        recovar_half1=_load_recovar_volume(paths["recovar_half1"]),
        recovar_half2=_load_recovar_volume(paths["recovar_half2"]),
        recovar_merged=_load_recovar_volume(paths["recovar_merged"]),
        relion_half1=_load_relion_volume(paths["relion_half1"]),
        relion_half2=_load_relion_volume(paths["relion_half2"]),
        relion_merged=_load_relion_volume(paths["relion_merged"]),
    )
    report["inputs"] = {
        label: {"path": str(path), "sha256": _sha256(path)}
        for label, path in paths.items()
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_shellwise_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez(args.output_shellwise_npz, **shellwise)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
