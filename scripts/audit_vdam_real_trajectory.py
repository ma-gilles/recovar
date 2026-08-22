#!/usr/bin/env python3
"""Audit a frozen real-particle K=1 InitialModel trajectory without pseudo-GT."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from scripts.audit_vdam_fsc_trajectory import (
        _artifact_paths,
        _load_json,
        _map_metric,
        _required_checkpoints,
        _validate_iteration_one_particle_subset,
        _validate_run_contract,
    )
    from scripts.summarize_em_completion_bench import _load_relion_volume
else:
    from audit_vdam_fsc_trajectory import (
        _artifact_paths,
        _load_json,
        _map_metric,
        _required_checkpoints,
        _validate_iteration_one_particle_subset,
        _validate_run_contract,
    )
    from summarize_em_completion_bench import _load_relion_volume


SCHEMA = "recovar.vdam_relion_real_data_trajectory_audit.v1"
SUITE_SCHEMA = "recovar.vdam_relion_real_data_suite.v1"


class RealDataAuditError(RuntimeError):
    """Raised when real-data parity evidence violates its frozen contract."""


def _case(scorecard: dict[str, Any], case_id: str) -> dict[str, Any]:
    if scorecard.get("schema") != SUITE_SCHEMA:
        raise RealDataAuditError(f"unsupported scorecard schema: {scorecard.get('schema')!r}")
    matches = [row for row in scorecard.get("cases", []) if row.get("id") == case_id]
    if len(matches) != 1:
        raise RealDataAuditError(f"expected one case {case_id}, found {len(matches)}")
    return matches[0]


def audit(
    *,
    scorecard_path: Path,
    case_id: str,
    recovar_dir: Path,
    relion_dir: Path,
    provenance_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    scorecard = _load_json(scorecard_path, label="real-data scorecard")
    case = _case(scorecard, case_id)
    definition = case["definition"]
    acceptance = scorecard["acceptance_contract"]
    checkpoints = _required_checkpoints(acceptance, definition)
    if acceptance.get("correlation_used") is not False:
        raise RealDataAuditError("real-data scorecard must explicitly forbid correlation")

    input_star = Path(case["input_star"])
    if hashlib.sha256(input_star.read_bytes()).hexdigest() != case["input_star_sha256"]:
        raise RealDataAuditError("input STAR digest differs from the frozen real-data suite")
    source_indices = input_star.with_name("source_indices.npy")
    if hashlib.sha256(source_indices.read_bytes()).hexdigest() != case["source_indices_sha256"]:
        raise RealDataAuditError("source-index digest differs from the frozen real-data suite")

    run_contract = _validate_run_contract(
        definition,
        _load_json(recovar_dir / "run_native_options.json", label="native options"),
        _load_json(relion_dir / "relion_command.json", label="RELION command"),
    )
    provenance = _load_json(provenance_path, label="run provenance")
    gpu_uuid = provenance.get("physical_gpu_uuid")
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        raise RealDataAuditError("run provenance has no physical GPU UUID")
    subset = _validate_iteration_one_particle_subset(input_star.parent, recovar_dir, relion_dir)

    shellwise: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    threshold = float(acceptance["cross_engine_fsc_auc_min"])
    for iteration in checkpoints:
        rec_paths = _artifact_paths(recovar_dir, iteration)
        rel_paths = _artifact_paths(relion_dir, iteration)
        missing = [str(path) for path in (*rec_paths.values(), *rel_paths.values()) if not path.is_file()]
        if missing:
            raise RealDataAuditError(f"iteration {iteration} is missing artifacts: {missing}")
        rec_map = _load_relion_volume(rec_paths["class001.mrc"])
        rel_map = _load_relion_volume(rel_paths["class001.mrc"])
        if rec_map.shape != rel_map.shape:
            raise RealDataAuditError(
                f"iteration {iteration} map shapes differ: {rec_map.shape} != {rel_map.shape}"
            )
        metric = _map_metric(rec_map, rel_map, key=f"it{iteration:03d}_cross_engine", shellwise=shellwise)
        rows.append(
            {
                "iteration": iteration,
                "cross_engine": metric,
                "pass": bool(metric["fsc_auc"] >= threshold),
                "artifact_topology_exact": True,
            }
        )

    report = {
        "schema": SCHEMA,
        "suite_id": scorecard["suite_id"],
        "case_id": case_id,
        "dataset": case["dataset"],
        "result": "pass" if all(row["pass"] for row in rows) else "fail",
        "metric_policy": "signed shellwise RECOVAR-vs-RELION FSC and normalized non-DC FSC-AUC only",
        "correlation_used": False,
        "thresholds": {"cross_engine_fsc_auc_min": threshold},
        "run_contract": run_contract,
        "same_physical_gpu": True,
        "physical_gpu_uuid": gpu_uuid,
        "iteration_one_particle_subset": subset,
        "checkpoints": rows,
        "minimum_cross_engine_fsc_auc": min(row["cross_engine"]["fsc_auc"] for row in rows),
        "runtime": {
            "relion_wall_s": provenance["relion_wall_s"],
            "recovar_wall_s": provenance["recovar_wall_s"],
            "recovar_over_relion": provenance["recovar_wall_s"] / provenance["relion_wall_s"],
        },
    }
    return report, shellwise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--recovar-dir", type=Path, required=True)
    parser.add_argument("--relion-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-shells-npz", type=Path, required=True)
    args = parser.parse_args(argv)
    report, shellwise = audit(
        scorecard_path=args.scorecard,
        case_id=args.case_id,
        recovar_dir=args.recovar_dir,
        relion_dir=args.relion_dir,
        provenance_path=args.provenance,
    )
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output_shells_npz, **shellwise)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
