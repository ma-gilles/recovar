#!/usr/bin/env python3
"""Seal repeatability of exact VDAM BPref host-replay hypotheses."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audit_vdam_repeat_panel import RepeatPanelError, _load_json

SCHEMA = "recovar.vdam_bpref_replay_repeat_panel.v1"
SUBMISSION_SCHEMA = "recovar.vdam_replay_repeat_panel_submission.v2"
REPORT_SCHEMA = "recovar.vdam_worker_private_host_replay.v6"
ARRAY_NAMES = ("reduced_real", "reduced_imag", "reduced_weight")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arm_directories(panel_root: Path, kind: str, repeat_count: int) -> list[Path]:
    arms = sorted(panel_root.glob(f"arm-*-{kind}"))
    if len(arms) != repeat_count:
        raise RepeatPanelError(
            f"expected {repeat_count} {kind} replay arms, found {len(arms)}"
        )
    return arms


def _load_arm(
    arm: Path,
    *,
    kind: str,
    science_head: str,
    physical_gpu_uuid: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    provenance = _load_json(arm / "provenance.json", label=f"{arm.name} provenance")
    report = _load_json(
        arm / "replay" / "worker_private_report.json",
        label=f"{arm.name} report",
    )
    accumulator_path = arm / "replay" / "worker_private_accumulators.npz"
    if not accumulator_path.is_file():
        raise RepeatPanelError(f"{arm.name}: accumulator artifact is missing")

    expected_shared = kind == "shared"
    if provenance.get("science_head") != science_head:
        raise RepeatPanelError(f"{arm.name}: science head differs")
    if provenance.get("gpu_uuid") != physical_gpu_uuid:
        raise RepeatPanelError(f"{arm.name}: physical GPU differs")
    if provenance.get("shared_accumulators") is not expected_shared:
        raise RepeatPanelError(f"{arm.name}: accumulator topology differs")
    if provenance.get("callbacks_merged_in_native_launch_order") is not True:
        raise RepeatPanelError(f"{arm.name}: native callback order was not replayed")
    if report.get("schema") != REPORT_SCHEMA or report.get("status") != "complete":
        raise RepeatPanelError(f"{arm.name}: replay report is incomplete or incompatible")
    if report.get("callbacks_merged_in_native_launch_order") is not True:
        raise RepeatPanelError(f"{arm.name}: report did not replay native callback order")
    if report.get("worker_private_accumulators") is expected_shared:
        raise RepeatPanelError(f"{arm.name}: report accumulator topology differs")
    if report.get("native_physical_grid_replayed") is not True:
        raise RepeatPanelError(f"{arm.name}: native physical grid was not replayed")
    if report.get("native_worker_owners_replayed") is not True:
        raise RepeatPanelError(f"{arm.name}: native worker owners were not replayed")
    if report.get("native_euler_panels_replayed") is not True:
        raise RepeatPanelError(f"{arm.name}: native Euler panels were not replayed")
    if int(report.get("worker_count", -1)) != 8 or int(report.get("callback_count", -1)) != 1:
        raise RepeatPanelError(f"{arm.name}: worker or callback topology differs")

    with np.load(accumulator_path, allow_pickle=False) as source:
        missing = sorted(set(ARRAY_NAMES) - set(source.files))
        if missing:
            raise RepeatPanelError(f"{arm.name}: accumulator arrays are missing: {missing}")
        arrays = {name: np.asarray(source[name]) for name in ARRAY_NAMES}
    for name, value in arrays.items():
        if value.dtype != np.float32 or value.size == 0 or not np.all(np.isfinite(value)):
            raise RepeatPanelError(f"{arm.name}: {name} is not finite nonempty float32")

    return (
        {
            "arm": arm.name,
            "job_id": str(provenance.get("job_id", "")),
            "gpu_uuid": str(provenance["gpu_uuid"]),
            "hypothesis": str(report.get("hypothesis", "")),
            "accumulator_sha256": _sha256(accumulator_path),
        },
        arrays,
    )


def _pairwise_metrics(arrays: list[np.ndarray]) -> dict[str, Any]:
    shapes = {value.shape for value in arrays}
    if len(shapes) != 1:
        raise RepeatPanelError(f"replay accumulator shapes differ: {shapes}")
    rows = []
    for (left_index, left), (right_index, right) in itertools.combinations(
        enumerate(arrays, start=1), 2
    ):
        denominator = float(np.linalg.norm(left.ravel()))
        if not np.isfinite(denominator) or denominator == 0.0:
            raise RepeatPanelError("replay accumulator has zero or nonfinite norm")
        difference = right - left
        rows.append(
            {
                "left_repeat": left_index,
                "right_repeat": right_index,
                "relative_l2": float(np.linalg.norm(difference.ravel()) / denominator),
                "max_absolute_error": float(np.max(np.abs(difference))),
                "exact_count": int(np.count_nonzero(left == right)),
                "value_count": int(left.size),
            }
        )
    if not rows:
        raise RepeatPanelError("replay repeat panel requires at least two arms per topology")
    values = [row["relative_l2"] for row in rows]
    return {
        "minimum_pairwise_relative_l2": min(values),
        "maximum_pairwise_relative_l2": max(values),
        "pairwise": rows,
    }


def audit_replay_repeat_panel(
    panel_root: Path,
    *,
    repeat_count: int = 4,
    expected_gpu_uuid: str | None = None,
) -> dict[str, Any]:
    panel_root = panel_root.resolve()
    if repeat_count < 2:
        raise RepeatPanelError("replay repeat panel requires at least two arms per topology")
    submission = _load_json(
        panel_root / "submission_provenance.json", label="submission provenance"
    )
    if submission.get("schema") != SUBMISSION_SCHEMA:
        raise RepeatPanelError("submission provenance schema differs")
    science_head = str(submission.get("science_head", ""))
    physical_gpu_uuid = str(submission.get("physical_gpu_uuid", ""))
    cuda_library_sha256 = str(submission.get("cuda_library_sha256", ""))
    if len(science_head) != 40 or len(cuda_library_sha256) != 64:
        raise RepeatPanelError("submission source or CUDA hash is invalid")
    if not physical_gpu_uuid.startswith("GPU-"):
        raise RepeatPanelError("submission physical GPU UUID is invalid")
    if expected_gpu_uuid is not None and physical_gpu_uuid != expected_gpu_uuid:
        raise RepeatPanelError(
            f"panel GPU {physical_gpu_uuid} differs from expected {expected_gpu_uuid}"
        )

    panels: dict[str, Any] = {}
    for kind in ("private", "shared"):
        arm_rows = []
        values = {name: [] for name in ARRAY_NAMES}
        for arm in _arm_directories(panel_root, kind, repeat_count):
            arm_row, arrays = _load_arm(
                arm,
                kind=kind,
                science_head=science_head,
                physical_gpu_uuid=physical_gpu_uuid,
            )
            arm_rows.append(arm_row)
            for name in ARRAY_NAMES:
                values[name].append(arrays[name])
        panels[kind] = {
            "repeat_count": repeat_count,
            "arms": arm_rows,
            "metrics": {name: _pairwise_metrics(values[name]) for name in ARRAY_NAMES},
        }

    diameter_ratios = {}
    for name in ARRAY_NAMES:
        private_diameter = panels["private"]["metrics"][name][
            "maximum_pairwise_relative_l2"
        ]
        shared_diameter = panels["shared"]["metrics"][name][
            "maximum_pairwise_relative_l2"
        ]
        diameter_ratios[name] = (
            float(shared_diameter / private_diameter)
            if private_diameter > 0.0
            else None
        )

    return {
        "schema": SCHEMA,
        "result": "complete",
        "scoring": False,
        "scope": (
            "diagnostic exact-PTX repeatability for native owner/count/Euler/order "
            "replay under worker-private and source-faithful shared BPref storage"
        ),
        "science_head": science_head,
        "physical_gpu_uuid": physical_gpu_uuid,
        "cuda_library_sha256": cuda_library_sha256,
        "repeat_count_per_topology": repeat_count,
        "panels": panels,
        "shared_over_private_maximum_diameter_ratio": diameter_ratios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", required=True, type=Path)
    parser.add_argument("--repeat-count", type=int, default=4)
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit_replay_repeat_panel(
        args.panel_root,
        repeat_count=args.repeat_count,
        expected_gpu_uuid=args.expected_gpu_uuid,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
