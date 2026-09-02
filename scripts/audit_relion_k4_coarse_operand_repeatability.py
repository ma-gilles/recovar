#!/usr/bin/env python3
"""Admit a K=4 coarse-operand observer only after repeated inertness gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "recovar.relion_k4_coarse_operand_repeatability.v1"
OPERAND_SCHEMA = "relion-k4-coarse-operand-validation-v1"
INERTNESS_SCHEMA = "recovar.relion_k4_coarse_score_capture_inertness.v1"
MINIMUM_REPEATS = 3


class AuditError(RuntimeError):
    """Raised when a repeat is incomplete or fails its fixed gate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, schema: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing report: {path}")
    report = json.loads(path.read_text())
    _require(report.get("schema") == schema, f"report schema differs: {path}")
    return report


def _job_identity(root: Path) -> tuple[int, Path, Path]:
    controls = tuple(sorted((root / "provenance").glob("scontrol_*.txt")))
    science = tuple(sorted((root / "provenance").glob("science_outputs_*.sha256")))
    _require(len(controls) == len(science) == 1, f"ambiguous provenance panel: {root}")
    match = re.fullmatch(r"scontrol_(\d+)\.txt", controls[0].name)
    _require(match is not None, f"invalid Slurm provenance name: {controls[0]}")
    job_id = int(match.group(1))
    _require(
        science[0].name == f"science_outputs_{job_id}.sha256" and science[0].stat().st_size > 0,
        f"science manifest differs from Slurm job identity: {root}",
    )
    text = controls[0].read_text()
    _require("ReqTRES=cpu=8,mem=192G,node=1,billing=15,gres/gpu=1" in text, f"requested resources differ: {root}")
    _require("AllocTRES=cpu=8,mem=192G,node=1,billing=15,gres/gpu=1" in text, f"allocated resources differ: {root}")
    return job_id, controls[0], science[0]


def build_report(run_roots: tuple[Path, ...]) -> dict[str, Any]:
    roots = tuple(Path(root).resolve() for root in run_roots)
    _require(len(roots) >= MINIMUM_REPEATS, "at least three paired repeats are required")
    _require(len(set(roots)) == len(roots), "repeat roots must be distinct")

    records: list[dict[str, Any]] = []
    topology: tuple[int, int, int, int] | None = None
    fixed_gates: dict[str, Any] | None = None
    for root in roots:
        _require(root.is_dir(), f"repeat root does not exist: {root}")
        operand_path = root / "analysis" / "operand_capture_validation.json"
        inertness_path = root / "analysis" / "capture_inertness.json"
        operand = _load(operand_path, OPERAND_SCHEMA)
        inertness = _load(inertness_path, INERTNESS_SCHEMA)
        _require(
            operand.get("status") == "pass" and operand.get("capture_ready") is True,
            f"operand validation failed: {root}",
        )
        _require(
            operand.get("fixed_metric", {}).get("evaluated_artifacts")
            == operand.get("fixed_metric", {}).get("passed_artifacts"),
            f"operand arithmetic replay is incomplete: {root}",
        )
        current_topology = (
            int(operand["particle_count"]),
            int(operand["class_count"]),
            int(operand["artifact_count"]),
            int(operand["iteration"]),
        )
        _require(current_topology == (16, 4, 64, 1), f"fixed K=4 panel differs: {root}")
        if topology is None:
            topology = current_topology
            fixed_gates = operand["fixed_gates"]
        _require(current_topology == topology, f"repeat topology changed: {root}")
        _require(operand["fixed_gates"] == fixed_gates, f"operand gates changed: {root}")

        strict = inertness.get("strict_gate", {})
        _require(inertness.get("status") == "pass", f"capture inertness failed: {root}")
        for field in (
            "capture_validated",
            "iteration_zero_numeric_state_exact",
            "iteration_one_decision_fields_exact",
            "iteration_one_maps_within_floor",
        ):
            _require(strict.get(field) is True, f"inertness gate {field} failed: {root}")
        iteration_one = next(
            (item for item in inertness.get("iterations", ()) if item.get("iteration") == 1),
            None,
        )
        _require(iteration_one is not None, f"iteration-1 inertness record is absent: {root}")
        maps = iteration_one.get("maps", ())
        _require(len(maps) == 4, f"class-map inertness panel differs: {root}")
        job_id, scontrol_path, science_path = _job_identity(root)
        records.append(
            {
                "run_root": str(root),
                "job_id": job_id,
                "operand_validation": {
                    "path": str(operand_path),
                    "sha256": _sha256(operand_path),
                },
                "capture_inertness": {
                    "path": str(inertness_path),
                    "sha256": _sha256(inertness_path),
                    "classification": inertness["classification"],
                },
                "provenance": {
                    "scontrol": str(scontrol_path),
                    "science_manifest": str(science_path),
                    "science_manifest_sha256": _sha256(science_path),
                },
                "class_map_fsc_auc_minimum": min(float(item["fsc_auc"]) for item in maps),
                "class_map_relative_l2_maximum": max(float(item["relative_l2"]) for item in maps),
            }
        )

    assert topology is not None and fixed_gates is not None
    return {
        "schema": SCHEMA,
        "status": "pass",
        "classification": "coarse_operand_capture_repeatably_inert_at_recorded_floor",
        "metric_policy": (
            "each independent control/instrumented repeat must pass the fixed operand "
            "arithmetic gate, preserve every recorded iteration-1 decision, and keep all "
            "four class maps above the predeclared FSC-AUC/relative-L2 inertness floor"
        ),
        "minimum_repeats": MINIMUM_REPEATS,
        "repeat_count": len(records),
        "fixed_panel": {
            "particle_count": topology[0],
            "class_count": topology[1],
            "artifact_count": topology[2],
            "iteration": topology[3],
            "operand_gates": fixed_gates,
        },
        "summary": {
            "passed_repeats": len(records),
            "class_map_fsc_auc_minimum": min(
                item["class_map_fsc_auc_minimum"] for item in records
            ),
            "class_map_relative_l2_maximum": max(
                item["class_map_relative_l2_maximum"] for item in records
            ),
            "iteration_one_decision_exact_repeats": len(records),
        },
        "repeats": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")
    report = build_report(tuple(args.run_root))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
