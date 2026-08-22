#!/usr/bin/env python3
"""Audit two complete K-class InitialModel pairs for native repeatability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from scripts.audit_vdam_kclass_trajectory import audit_trajectory
else:
    from audit_vdam_kclass_trajectory import audit_trajectory


SCHEMA = "recovar.vdam_kclass_repeatability.v1"


class RepeatabilityError(RuntimeError):
    """Raised when paired repeatability evidence is incomplete or mixed."""


def _load_pair_report(pair_root: Path) -> dict[str, Any]:
    path = pair_root / "pair_report.json"
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RepeatabilityError(f"cannot read pair report {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise RepeatabilityError(f"pair report must contain a JSON object: {path}")
    return report


def _validated_contract(pair_roots: tuple[Path, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reports = [_load_pair_report(root) for root in pair_roots]
    if any(report.get("schema") != "recovar.vdam_kclass_pair.v1" for report in reports):
        raise RepeatabilityError("repeat inputs do not use the K-class pair schema")
    if any(bool(report.get("git_dirty")) for report in reports):
        raise RepeatabilityError("repeatability qualification requires clean source trees")
    if any(report.get("audit", {}).get("result") != "pass" for report in reports):
        raise RepeatabilityError("each repeat must independently pass cross-engine parity")

    audit_contracts = []
    for report in reports:
        audit = report.get("audit", {})
        thresholds = audit.get("thresholds", {})
        audit_contracts.append(
            {
                "K": int(audit.get("K", 0)),
                "checkpoints": tuple(int(value) for value in audit.get("checkpoints", ())),
                "minimum_fsc_auc": float(thresholds.get("minimum_per_class_fsc_auc", np.nan)),
                "minimum_assignment_accuracy": float(
                    thresholds.get("minimum_class_assignment_accuracy", np.nan)
                ),
                "git_head": str(report.get("git_head", "")),
                "physical_gpu_uuid": str(report.get("physical_gpu_uuid", "")),
                "relion_sha256": str(report.get("relion_sha256", "")),
                "fixture_dir": str(report.get("fixture_dir", "")),
                "fixture_sha256": report.get("fixture_sha256"),
            }
        )
    if audit_contracts[0] != audit_contracts[1]:
        raise RepeatabilityError("repeat pair contracts, source, GPU, executable, or fixtures differ")
    contract = audit_contracts[0]
    if contract["K"] < 2:
        raise RepeatabilityError("K-class repeatability requires K >= 2")
    if not contract["checkpoints"] or contract["checkpoints"] != tuple(
        range(contract["checkpoints"][-1] + 1)
    ):
        raise RepeatabilityError("repeat pairs must audit every written iteration")
    if not np.isfinite(contract["minimum_fsc_auc"]) or not np.isfinite(
        contract["minimum_assignment_accuracy"]
    ):
        raise RepeatabilityError("repeat pair thresholds must be finite")
    if len(contract["git_head"]) != 40:
        raise RepeatabilityError("repeat pair source head is invalid")
    if not contract["physical_gpu_uuid"].startswith("GPU-"):
        raise RepeatabilityError("repeat pair physical GPU identity is invalid")
    if len(contract["relion_sha256"]) != 64:
        raise RepeatabilityError("repeat pair RELION hash is invalid")
    return reports, contract


def audit_repeatability(
    *, pair_roots: tuple[Path, Path]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    reports, contract = _validated_contract(pair_roots)
    audit_kwargs = {
        "K": contract["K"],
        "checkpoints": contract["checkpoints"],
        "minimum_fsc_auc": contract["minimum_fsc_auc"],
        "minimum_assignment_accuracy": contract["minimum_assignment_accuracy"],
    }
    recovar_report, recovar_shells = audit_trajectory(
        candidate_dir=pair_roots[1] / "recovar",
        reference_dir=pair_roots[0] / "recovar",
        **audit_kwargs,
    )
    relion_report, relion_shells = audit_trajectory(
        candidate_dir=pair_roots[1] / "relion",
        reference_dir=pair_roots[0] / "relion",
        **audit_kwargs,
    )
    result = (
        "pass"
        if recovar_report["result"] == "pass"
        and relion_report["result"] == "pass"
        and all(report["audit"]["result"] == "pass" for report in reports)
        else "fail"
    )
    report = {
        "schema": SCHEMA,
        "result": result,
        "K": contract["K"],
        "checkpoints": list(contract["checkpoints"]),
        "thresholds": {
            "minimum_per_class_fsc_auc": contract["minimum_fsc_auc"],
            "minimum_class_assignment_accuracy": contract["minimum_assignment_accuracy"],
        },
        "git_head": contract["git_head"],
        "physical_gpu_uuid": contract["physical_gpu_uuid"],
        "relion_sha256": contract["relion_sha256"],
        "fixture_dir": contract["fixture_dir"],
        "pair_roots": [str(root) for root in pair_roots],
        "individual_cross_engine_results": [item["audit"]["result"] for item in reports],
        "recovar_repeat_audit": recovar_report,
        "relion_repeat_audit": relion_report,
        "metric_policy": "permutation-invariant signed shellwise FSC-AUC and hard-class agreement; no correlation",
        "correlation_used": False,
    }
    shellwise = {
        **{f"recovar_repeat_{key}": value for key, value in recovar_shells.items()},
        **{f"relion_repeat_{key}": value for key, value in relion_shells.items()},
    }
    return report, shellwise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-pair-root", type=Path, required=True)
    parser.add_argument("--second-pair-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-shells-npz", type=Path, required=True)
    args = parser.parse_args(argv)
    report, shellwise = audit_repeatability(
        pair_roots=(args.first_pair_root.resolve(), args.second_pair_root.resolve())
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output_shells_npz, **shellwise)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
