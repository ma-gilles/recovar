#!/usr/bin/env python3
"""Audit every candidate state trajectory against a same-GPU native panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from recovar.data_io.starfile import read_star
from scripts.audit_vdam_candidate_state_envelope import (
    POSE_TOLERANCE_DEG,
    TRANSLATION_TOLERANCE_ANGST,
    classify_schedule_mode_envelope,
    compare_particle_tables_to_native_set,
)
from scripts.audit_vdam_repeat_panel import TRAJECTORY_SCHEMA, RepeatPanelError, _load_json
from scripts.audit_vdam_sampling_trajectory import audit_sampling_trajectory

SCHEMA = "recovar.vdam_repeat_state_panel.v1"


def _column(table, name: str) -> str:
    matches = [str(column) for column in table.columns if str(column).lstrip("_") == name]
    if len(matches) != 1:
        raise RepeatPanelError(f"expected one {name} column, found {matches}")
    return matches[0]


def _pixel_size(root: Path) -> float:
    _, optics = read_star(str(root / "data" / "particles.star"))
    if optics is None or len(optics) == 0:
        raise RepeatPanelError(f"{root}: fixture optics table is missing")
    column = _column(optics, "rlnImagePixelSize")
    values = sorted(set(float(value) for value in optics[column].tolist()))
    if len(values) != 1 or not np.isfinite(values[0]) or values[0] <= 0.0:
        raise RepeatPanelError(f"{root}: expected one positive pixel size, found {values}")
    return values[0]


def _validated_roots(
    *, scorecard: dict[str, Any], case_id: str, panel_root: Path, repeat_count: int
) -> tuple[list[Path], tuple[int, ...], dict[str, Any]]:
    if repeat_count < 2:
        raise RepeatPanelError("state repeat panel requires at least two paired repeats")
    matches = [row for row in scorecard.get("cases", ()) if row.get("id") == case_id]
    if len(matches) != 1:
        raise RepeatPanelError(f"expected one scorecard row for {case_id}")
    checkpoints = tuple(
        int(value) for value in scorecard["acceptance_contract"]["required_checkpoints"]
    )
    if not checkpoints or checkpoints[0] != 0 or checkpoints != tuple(sorted(set(checkpoints))):
        raise RepeatPanelError("scorecard checkpoints must be sorted, unique, and start at zero")

    roots = [
        (panel_root / f"repeat-{index:02d}" / case_id).resolve()
        for index in range(1, repeat_count + 1)
    ]
    source_heads: set[str] = set()
    gpu_uuids: set[str] = set()
    relion_hashes: set[str] = set()
    cuda_hashes: set[str] = set()
    for index, root in enumerate(roots, start=1):
        trajectory = _load_json(root / "trajectory_audit.json", label=f"repeat {index} audit")
        provenance = _load_json(root / "run_provenance.json", label=f"repeat {index} provenance")
        gpu = _load_json(root / "paired_gpu_uuid.json", label=f"repeat {index} GPU report")
        if trajectory.get("schema") != TRAJECTORY_SCHEMA:
            raise RepeatPanelError(f"repeat {index}: trajectory schema differs")
        if trajectory.get("suite_id") != scorecard.get("suite_id") or trajectory.get(
            "case_id"
        ) != case_id:
            raise RepeatPanelError(f"repeat {index}: suite or case identity differs")
        observed = tuple(int(row["iteration"]) for row in trajectory.get("checkpoints", ()))
        if observed != checkpoints or not bool(trajectory.get("artifact_topology_exact")):
            raise RepeatPanelError(f"repeat {index}: checkpoint topology differs")
        physical = str(gpu.get("physical_gpu_uuid", ""))
        if {
            physical,
            str(gpu.get("relion_gpu_uuid", "")),
            str(gpu.get("recovar_gpu_uuid", "")),
        } != {physical} or not physical.startswith("GPU-"):
            raise RepeatPanelError(f"repeat {index}: paired GPU identity differs")
        source_heads.add(str(provenance.get("git_head", "")))
        gpu_uuids.add(physical)
        relion_hashes.add(str(provenance.get("relion_reference", {}).get("executable_sha256", "")))
        cuda_hashes.add(
            str(
                provenance.get("recovar_native_extensions", {})
                .get("cuda_backproject", {})
                .get("sha256", "")
            )
        )
    if len(source_heads) != 1 or any(len(value) != 40 for value in source_heads):
        raise RepeatPanelError(f"repeat panel contains mixed or invalid source heads: {source_heads}")
    if len(gpu_uuids) != 1:
        raise RepeatPanelError(f"repeat panel spans physical GPUs: {gpu_uuids}")
    if len(relion_hashes) != 1 or any(len(value) != 64 for value in relion_hashes):
        raise RepeatPanelError("repeat panel contains mixed or invalid RELION executable hashes")
    if len(cuda_hashes) != 1 or any(len(value) != 64 for value in cuda_hashes):
        raise RepeatPanelError("repeat panel contains mixed or invalid CUDA library hashes")
    return roots, checkpoints, {
        "source_head": next(iter(source_heads)),
        "physical_gpu_uuid": next(iter(gpu_uuids)),
        "relion_executable_sha256": next(iter(relion_hashes)),
        "cuda_library_sha256": next(iter(cuda_hashes)),
    }


def audit_state_panel(
    *, scorecard_path: Path, case_id: str, panel_root: Path, repeat_count: int
) -> dict[str, Any]:
    scorecard = _load_json(scorecard_path.resolve(), label="scorecard")
    roots, checkpoints, provenance = _validated_roots(
        scorecard=scorecard,
        case_id=case_id,
        panel_root=panel_root.resolve(),
        repeat_count=repeat_count,
    )
    iterations = tuple(value for value in checkpoints if value > 0)
    pixel_sizes = [_pixel_size(root) for root in roots]
    if len(set(pixel_sizes)) != 1:
        raise RepeatPanelError(f"repeat fixtures have mixed pixel sizes: {pixel_sizes}")
    pixel_size = pixel_sizes[0]

    native_tables = {
        iteration: [
            read_star(str(root / "relion" / f"run_it{iteration:03d}_data.star"))[0]
            for root in roots
        ]
        for iteration in iterations
    }
    candidate_rows = []
    for candidate_index, root in enumerate(roots, start=1):
        fixture, _ = read_star(str(root / "data" / "particles.star"))
        identity_column = _column(fixture, "rlnImageName")
        sampling_reports = [
            audit_sampling_trajectory(
                root / "recovar",
                native_root / "relion",
                pixel_size=pixel_size,
                iterations=list(iterations),
            )
            for native_root in roots
        ]
        particle_checkpoints = []
        schedule_checkpoints = []
        for offset, iteration in enumerate(iterations):
            meta = _load_json(
                root / "recovar" / f"run_it{iteration:03d}_recovar_meta.json",
                label=f"candidate {candidate_index} iteration {iteration} metadata",
            )
            selected = np.asarray(meta.get("selected_particle_ids", ()), dtype=np.int64)
            if selected.size == 0 or np.any(selected < 0) or np.any(selected >= len(fixture)):
                raise RepeatPanelError(
                    f"candidate {candidate_index} iteration {iteration}: selected ids are invalid"
                )
            active_ids = set(fixture.iloc[selected][identity_column].astype(str).tolist())
            if len(active_ids) != selected.size:
                raise RepeatPanelError(
                    f"candidate {candidate_index} iteration {iteration}: selected ids are not unique"
                )
            candidate_table = read_star(
                str(root / "recovar" / f"run_it{iteration:03d}_data.star")
            )[0]
            particle_checkpoints.append(
                {
                    "iteration": iteration,
                    **compare_particle_tables_to_native_set(
                        candidate_table,
                        native_tables[iteration],
                        active_image_ids=active_ids,
                    ),
                }
            )
            schedule_checkpoints.append(
                {
                    "iteration": iteration,
                    **classify_schedule_mode_envelope(
                        [report["iterations"][offset] for report in sampling_reports]
                    ),
                }
            )
        particle_pass = all(row["pass"] for row in particle_checkpoints)
        schedule_pass = all(row["pass"] for row in schedule_checkpoints)
        candidate_rows.append(
            {
                "repeat_index": candidate_index,
                "result": "pass" if particle_pass and schedule_pass else "fail",
                "particle_result": "pass" if particle_pass else "fail",
                "schedule_result": "pass" if schedule_pass else "fail",
                "first_particle_failure_iteration": next(
                    (row["iteration"] for row in particle_checkpoints if not row["pass"]), None
                ),
                "first_schedule_failure_iteration": next(
                    (row["iteration"] for row in schedule_checkpoints if not row["pass"]), None
                ),
                "particle_checkpoints": particle_checkpoints,
                "schedule_checkpoints": schedule_checkpoints,
            }
        )

    result = "pass" if all(row["result"] == "pass" for row in candidate_rows) else "fail"
    return {
        "schema": SCHEMA,
        "suite_id": scorecard["suite_id"],
        "case_id": case_id,
        "result": result,
        "scoring": False,
        "scope": (
            "fixed-tolerance active-particle coverage by any native repeat and complete "
            "per-iteration native schedule-mode coverage for every candidate repeat"
        ),
        "strict_point_reference_results_preserved": True,
        "repeat_count": repeat_count,
        "checkpoints": list(checkpoints),
        "pixel_size": pixel_size,
        "thresholds": {
            "pose_tolerance_deg": POSE_TOLERANCE_DEG,
            "translation_tolerance_angst": TRANSLATION_TOLERANCE_ANGST,
        },
        "provenance": provenance,
        "candidate_repeats": candidate_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--repeat-count", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_state_panel(
        scorecard_path=args.scorecard,
        case_id=args.case_id,
        panel_root=args.panel_root,
        repeat_count=args.repeat_count,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
