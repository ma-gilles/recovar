#!/usr/bin/env python3
"""Compare K=1 case-7 operand arms at the fixed iteration-2 population boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_arm(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("arms must use LABEL=/absolute/path/to/particle.json")
    path = Path(raw_path)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("particle report paths must be absolute")
    return label, path


def _load_report(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    report = json.loads(path.read_text())
    iterations = report.get("iterations", [])
    if len(iterations) != 1 or int(iterations[0]["relion_iteration"]) != 2:
        raise ValueError(f"{path} is not a single physical-iteration-2 audit")
    arrays_path = Path(report["artifacts"]["compact_npz"]["path"])
    if not arrays_path.is_absolute() or not arrays_path.is_file():
        raise ValueError(f"invalid compact array artifact in {path}: {arrays_path}")
    with np.load(arrays_path, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    if str(arrays["schema"]) != "em_particle_state_distribution_arrays_v1":
        raise ValueError(f"unexpected compact schema in {arrays_path}: {arrays['schema']}")
    return report, arrays


def _metrics(report: dict) -> dict[str, float | int]:
    comparison = report["iterations"][0]["recovar_vs_relion"]
    return {
        "pmax_rmse": float(comparison["pmax"]["absolute"]["rmse"]),
        "pmax_p95_abs": float(comparison["pmax"]["absolute"]["p95"]),
        "support_mismatch_count": int(comparison["significant_support"]["different_count"]),
        "angular_error_rmse_deg": float(comparison["angular_error_deg"]["rmse"]),
        "translation_error_rmse_angstrom": float(comparison["translation_error"]["rmse"]),
    }


def _target_state(arrays: dict[str, np.ndarray], original_index: int) -> dict[str, float | int]:
    positions = np.flatnonzero(arrays["identity_row_index"] == original_index)
    if positions.size != 1:
        raise ValueError(f"original row {original_index} occurs {positions.size} times")
    index = int(positions[0])
    return {
        "pmax_recovar": float(arrays["it002_pmax_recovar"][index]),
        "pmax_relion": float(arrays["it002_pmax_relion"][index]),
        "support_recovar": int(arrays["it002_support_recovar"][index]),
        "support_relion": int(arrays["it002_support_relion"][index]),
        "angular_error_deg": float(arrays["it002_rotation_geodesic_deg"][index]),
        "translation_error_angstrom": float(arrays["it002_translation_l2"][index]),
    }


def _classify(control: dict, treatment: dict) -> tuple[str, dict[str, float | int | bool]]:
    pmax_improvement = 1.0 - treatment["pmax_rmse"] / control["pmax_rmse"]
    support_improvement = control["support_mismatch_count"] - treatment["support_mismatch_count"]
    regressions = {
        "pmax": bool(treatment["pmax_rmse"] > control["pmax_rmse"] * (1.0 + 1e-6)),
        "support": bool(treatment["support_mismatch_count"] > control["support_mismatch_count"]),
        "angular": bool(treatment["angular_error_rmse_deg"] > control["angular_error_rmse_deg"] + 1e-8),
        "translation": bool(
            treatment["translation_error_rmse_angstrom"] > control["translation_error_rmse_angstrom"] + 1e-8
        ),
    }
    if any(regressions.values()):
        classification = "rejected_population_regression"
    elif pmax_improvement > 0.5 and support_improvement >= 2:
        classification = "strong_population_improvement"
    elif pmax_improvement > 0.0 or support_improvement > 0:
        classification = "partial_population_improvement"
    else:
        classification = "no_material_population_change"
    effects = {
        "pmax_rmse_fractional_improvement": pmax_improvement,
        "support_mismatch_count_improvement": support_improvement,
        "angular_error_rmse_delta_deg": (treatment["angular_error_rmse_deg"] - control["angular_error_rmse_deg"]),
        "translation_error_rmse_delta_angstrom": (
            treatment["translation_error_rmse_angstrom"] - control["translation_error_rmse_angstrom"]
        ),
        "population_regressions": regressions,
    }
    return classification, effects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--arm", action="append", type=_parse_arm, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.control.is_absolute() or not args.output.is_absolute():
        parser.error("--control and --output must be absolute paths")

    control_report, control_arrays = _load_report(args.control)
    control_metrics = _metrics(control_report)
    identity_sha256 = str(control_arrays["identity_sha256"])
    control_support_ids = control_arrays["identity_row_index"][control_arrays["it002_support_delta"] != 0]
    fixed_target_ids = tuple(int(index) for index in control_support_ids)
    record = {
        "schema": "recovar.em.k1_case07.operand_population.v1",
        "status": "complete",
        "fixed_case": {"case": "k1-07", "particle_count": 100000, "physical_iteration": 2},
        "metric_policy": "source-ID-aligned Pmax/support/pose metrics; no correlation",
        "classification_policy": {
            "reject": "any Pmax, support-count, angular-RMSE, or translation-RMSE population regression",
            "strong": "no regression, greater-than-50% Pmax-RMSE improvement, and at least two fewer support mismatches",
            "partial": "no regression and any Pmax-RMSE or support-count improvement",
        },
        "identity_sha256": identity_sha256,
        "control": {
            "report": str(args.control),
            "report_sha256": _sha256(args.control),
            "metrics": control_metrics,
            "support_mismatch_original_indices": control_support_ids.tolist(),
            "targets": {str(index): _target_state(control_arrays, index) for index in fixed_target_ids},
        },
        "arms": {},
    }
    for label, path in args.arm:
        report, arrays = _load_report(path)
        if str(arrays["identity_sha256"]) != identity_sha256:
            raise ValueError(f"identity mismatch for arm {label}: {path}")
        metrics = _metrics(report)
        classification, effects = _classify(control_metrics, metrics)
        support_ids = arrays["identity_row_index"][arrays["it002_support_delta"] != 0]
        record["arms"][label] = {
            "report": str(path),
            "report_sha256": _sha256(path),
            "metrics": metrics,
            "effects_vs_control": effects,
            "classification": classification,
            "support_mismatch_original_indices": support_ids.tolist(),
            "targets": {str(index): _target_state(arrays, index) for index in fixed_target_ids},
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
