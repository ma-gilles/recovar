#!/usr/bin/env python3
"""Audit a same-GPU panel against stock RELION's native repeat envelope."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from scripts.summarize_em_completion_bench import _load_relion_volume, normalized_fsc_auc, shell_fsc
else:
    from summarize_em_completion_bench import _load_relion_volume, normalized_fsc_auc, shell_fsc


SCHEMA = "recovar.vdam_relion_repeat_panel.v1"
TRAJECTORY_SCHEMA = "recovar.vdam_relion_fsc_trajectory_audit.v1"


class RepeatPanelError(RuntimeError):
    """Raised when repeat-panel evidence is incomplete or mixed."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RepeatPanelError(f"cannot read {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RepeatPanelError(f"{label} must contain a JSON object: {path}")
    return value


def classify_checkpoint(
    *,
    relion_self_fsc_auc: list[float],
    recovar_self_fsc_auc: list[float],
    cross_engine_fsc_auc: list[float],
    gt_deltas: list[float],
    cross_engine_min: float,
    gt_delta_min: float,
) -> dict[str, Any]:
    """Classify parity without widening the frozen point thresholds."""

    if not relion_self_fsc_auc or not recovar_self_fsc_auc or not cross_engine_fsc_auc:
        raise RepeatPanelError("repeat classification requires native, candidate, and cross-engine comparisons")
    if not gt_deltas:
        raise RepeatPanelError("repeat classification requires per-run GT nondegradation evidence")
    native_floor = float(min(relion_self_fsc_auc))
    candidate_floor = float(min(recovar_self_fsc_auc))
    cross_floor = float(min(cross_engine_fsc_auc))
    cross_ceiling = float(max(cross_engine_fsc_auc))
    gt_floor = float(min(gt_deltas))
    checks = {
        "candidate_repeat_meets_frozen_cross_gate": candidate_floor >= float(cross_engine_min),
        "cross_engine_panel_reaches_frozen_cross_gate": cross_ceiling >= float(cross_engine_min),
        "worst_cross_engine_meets_frozen_gate_or_native_repeat_floor": (
            cross_floor >= float(cross_engine_min) or cross_floor >= native_floor
        ),
        "all_runs_meet_frozen_gt_nondegradation_gate": gt_floor >= float(gt_delta_min),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "native_relion_repeat_floor_fsc_auc": native_floor,
        "recovar_repeat_floor_fsc_auc": candidate_floor,
        "cross_engine_floor_fsc_auc": cross_floor,
        "cross_engine_ceiling_fsc_auc": cross_ceiling,
        "minimum_recovar_minus_relion_gt_fsc_auc": gt_floor,
    }


def _metric(lhs: np.ndarray, rhs: np.ndarray, *, key: str, shellwise: dict[str, np.ndarray]) -> float:
    curve = np.asarray(shell_fsc(lhs, rhs), dtype=np.float64)
    if curve.size <= 1 or not np.any(np.isfinite(curve[1:])):
        raise RepeatPanelError(f"{key} produced no finite non-DC FSC shells")
    shellwise[key] = curve
    value = float(normalized_fsc_auc(curve))
    if not np.isfinite(value):
        raise RepeatPanelError(f"{key} produced a non-finite FSC-AUC")
    return value


def _map_path(repeat_root: Path, engine: str, iteration: int) -> Path:
    return repeat_root / engine / f"run_it{iteration:03d}_class001.mrc"


def audit_repeat_panel(
    *,
    scorecard_path: Path,
    case_id: str,
    panel_root: Path,
    repeat_count: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if repeat_count < 2:
        raise RepeatPanelError("repeat_count must be at least two")
    scorecard = _load_json(scorecard_path, label="scorecard")
    matches = [case for case in scorecard.get("cases", []) if case.get("id") == case_id]
    if len(matches) != 1:
        raise RepeatPanelError(f"expected one scorecard row for {case_id}, found {len(matches)}")
    case = matches[0]
    acceptance = scorecard["acceptance_contract"]
    checkpoints = tuple(int(value) for value in acceptance["required_checkpoints"])
    cross_min = float(acceptance["cross_engine_fsc_auc_min"])
    gt_min = float(acceptance["recovar_minus_relion_gt_fsc_auc_min"])

    repeats: list[dict[str, Any]] = []
    source_heads: set[str] = set()
    gpu_uuids: set[str] = set()
    relion_hashes: set[str] = set()
    for repeat_index in range(1, repeat_count + 1):
        root = panel_root / f"repeat-{repeat_index:02d}" / case_id
        trajectory = _load_json(root / "trajectory_audit.json", label="trajectory audit")
        provenance = _load_json(root / "run_provenance.json", label="run provenance")
        gpu = _load_json(root / "paired_gpu_uuid.json", label="paired GPU report")
        if trajectory.get("schema") != TRAJECTORY_SCHEMA or trajectory.get("case_id") != case_id:
            raise RepeatPanelError(f"repeat {repeat_index}: trajectory schema or case identity differs")
        if trajectory.get("suite_id") != scorecard.get("suite_id"):
            raise RepeatPanelError(f"repeat {repeat_index}: suite identity differs")
        if not bool(trajectory.get("artifact_topology_exact")):
            raise RepeatPanelError(f"repeat {repeat_index}: artifact topology differs")
        observed_checkpoints = tuple(int(row["iteration"]) for row in trajectory.get("checkpoints", ()))
        if observed_checkpoints != checkpoints:
            raise RepeatPanelError(f"repeat {repeat_index}: checkpoint topology differs")
        physical = str(gpu.get("physical_gpu_uuid", ""))
        if {
            physical,
            str(gpu.get("relion_gpu_uuid", "")),
            str(gpu.get("recovar_gpu_uuid", "")),
        } != {physical} or not physical.startswith("GPU-"):
            raise RepeatPanelError(f"repeat {repeat_index}: paired physical GPU identity differs")
        source_heads.add(str(provenance.get("git_head", "")))
        gpu_uuids.add(physical)
        relion_hashes.add(str(provenance.get("relion_reference", {}).get("executable_sha256", "")))
        repeats.append({"index": repeat_index, "root": root, "trajectory": trajectory})
    if len(source_heads) != 1 or any(len(value) != 40 for value in source_heads):
        raise RepeatPanelError(f"repeat panel contains mixed or invalid source heads: {sorted(source_heads)}")
    if len(gpu_uuids) != 1:
        raise RepeatPanelError(f"repeat panel did not stay on one physical GPU: {sorted(gpu_uuids)}")
    if len(relion_hashes) != 1 or any(len(value) != 64 for value in relion_hashes):
        raise RepeatPanelError("repeat panel contains mixed or invalid RELION executable hashes")

    shellwise: dict[str, np.ndarray] = {}
    checkpoint_rows = []
    for iteration in checkpoints:
        relion_volumes = [
            _load_relion_volume(_map_path(repeat["root"], "relion", iteration)) for repeat in repeats
        ]
        recovar_volumes = [
            _load_relion_volume(_map_path(repeat["root"], "recovar", iteration)) for repeat in repeats
        ]
        relion_self = []
        recovar_self = []
        for lhs, rhs in itertools.combinations(range(repeat_count), 2):
            relion_self.append(
                _metric(
                    relion_volumes[lhs],
                    relion_volumes[rhs],
                    key=f"it{iteration:03d}_relion_r{lhs + 1:02d}_r{rhs + 1:02d}",
                    shellwise=shellwise,
                )
            )
            recovar_self.append(
                _metric(
                    recovar_volumes[lhs],
                    recovar_volumes[rhs],
                    key=f"it{iteration:03d}_recovar_r{lhs + 1:02d}_r{rhs + 1:02d}",
                    shellwise=shellwise,
                )
            )
        cross = []
        for rec_index, recovar_volume in enumerate(recovar_volumes):
            for rel_index, relion_volume in enumerate(relion_volumes):
                cross.append(
                    _metric(
                        recovar_volume,
                        relion_volume,
                        key=f"it{iteration:03d}_recovar_r{rec_index + 1:02d}_relion_r{rel_index + 1:02d}",
                        shellwise=shellwise,
                    )
                )
        gt_deltas = []
        for repeat in repeats:
            row = next(
                item for item in repeat["trajectory"]["checkpoints"] if int(item["iteration"]) == iteration
            )
            gt_deltas.append(float(row["recovar_minus_relion_gt_fsc_auc"]))
        classification = classify_checkpoint(
            relion_self_fsc_auc=relion_self,
            recovar_self_fsc_auc=recovar_self,
            cross_engine_fsc_auc=cross,
            gt_deltas=gt_deltas,
            cross_engine_min=cross_min,
            gt_delta_min=gt_min,
        )
        checkpoint_rows.append({"iteration": iteration, **classification})

    result = "pass" if all(row["pass"] for row in checkpoint_rows) else "fail"
    report = {
        "schema": SCHEMA,
        "suite_id": scorecard["suite_id"],
        "case_id": case_id,
        "case_name": case.get("name"),
        "result": result,
        "repeat_count": repeat_count,
        "source_head": next(iter(source_heads)),
        "physical_gpu_uuid": next(iter(gpu_uuids)),
        "relion_executable_sha256": next(iter(relion_hashes)),
        "thresholds": {
            "cross_engine_fsc_auc_min": cross_min,
            "recovar_minus_relion_gt_fsc_auc_min": gt_min,
        },
        "metric_policy": (
            "signed shellwise FSC and normalized non-DC FSC-AUC only; the frozen point gates are retained, "
            "and a cross-engine repeat panel must be no worse than the observed same-GPU native RELION floor"
        ),
        "correlation_used": False,
        "individual_results": [repeat["trajectory"]["result"] for repeat in repeats],
        "checkpoints": checkpoint_rows,
    }
    return report, shellwise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--repeat-count", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-shells-npz", type=Path, required=True)
    args = parser.parse_args(argv)
    report, shellwise = audit_repeat_panel(
        scorecard_path=args.scorecard.resolve(),
        case_id=args.case_id,
        panel_root=args.panel_root.resolve(),
        repeat_count=int(args.repeat_count),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output_shells_npz, **shellwise)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
