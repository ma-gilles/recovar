#!/usr/bin/env python3
"""Compare matched native VDAM BPref repeat width with exact host replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_vdam_mstep_boundary import _read_relion_array
from scripts.audit_vdam_bpref_replay_repeat_panel import (
    _pairwise_metrics,
    audit_replay_repeat_panel,
)
from scripts.audit_vdam_repeat_panel import RepeatPanelError, _load_json

SCHEMA = "recovar.vdam_native_bpref_repeat_panel.v1"
NATIVE_PROVENANCE_SCHEMA = "recovar.vdam_native_bpref_repeat.v1"
FIELDS = (
    "data_half0",
    "weight_half0",
    "data_half1",
    "weight_half1",
)


def _native_filenames(iteration: int) -> dict[str, tuple[str, bool]]:
    token = f"it{int(iteration)}"
    return {
        "data_half0": (f"pipe_{token}_c0_bp_data_pre_reweight.bin", True),
        "weight_half0": (f"pipe_{token}_c0_bp_weight.bin", False),
        "data_half1": (f"pipe_{token}_c0_bp_data_h_pre_reweight.bin", True),
        "weight_half1": (f"pipe_{token}_c0_bp_weight_h.bin", False),
    }


def _native_arms(panel_root: Path, repeat_count: int) -> list[Path]:
    arms = sorted(panel_root.glob("repeat-*"))
    if len(arms) != repeat_count:
        raise RepeatPanelError(
            f"expected {repeat_count} native BPref repeats, found {len(arms)}"
        )
    return arms


def _load_native_panel(
    panel_root: Path,
    *,
    repeat_count: int,
    iteration: int,
) -> tuple[list[dict[str, Any]], dict[str, list[np.ndarray]], dict[str, str]]:
    rows = []
    values = {name: [] for name in FIELDS}
    source_heads: set[str] = set()
    gpu_uuids: set[str] = set()
    filenames = _native_filenames(iteration)
    for arm in _native_arms(panel_root, repeat_count):
        provenance_path = arm / "provenance" / "native_only_completion.json"
        provenance = _load_json(provenance_path, label=f"{arm.name} provenance")
        if provenance.get("schema") != NATIVE_PROVENANCE_SCHEMA:
            raise RepeatPanelError(f"{arm.name}: native-only provenance schema differs")
        if int(provenance.get("iteration", -1)) != iteration:
            raise RepeatPanelError(f"{arm.name}: native-only iteration differs")
        if not (arm / "NATIVE_ONLY_SUCCESS").is_file():
            raise RepeatPanelError(f"{arm.name}: native-only success seal is missing")
        if not (arm / "provenance" / "native_only_evidence.sha256").is_file():
            raise RepeatPanelError(f"{arm.name}: native-only evidence seal is missing")
        source_heads.add(str(provenance.get("git_head", "")))
        gpu_uuids.add(str(provenance.get("physical_gpu_uuid", "")))
        for name, (filename, complex_values) in filenames.items():
            values[name].append(
                _read_relion_array(
                    arm / "native_mstep" / filename,
                    complex_values=complex_values,
                )
            )
        rows.append(
            {
                "arm": arm.name,
                "job_id": str(provenance.get("job_id", "")),
                "science_head": str(provenance.get("git_head", "")),
                "gpu_uuid": str(provenance.get("physical_gpu_uuid", "")),
            }
        )
    if len(source_heads) != 1 or any(len(value) != 40 for value in source_heads):
        raise RepeatPanelError(f"native BPref panel contains mixed source heads: {source_heads}")
    if len(gpu_uuids) != 1 or not next(iter(gpu_uuids)).startswith("GPU-"):
        raise RepeatPanelError(f"native BPref panel spans physical GPUs: {gpu_uuids}")
    return rows, values, {
        "science_head": next(iter(source_heads)),
        "physical_gpu_uuid": next(iter(gpu_uuids)),
    }


def _load_replay_values(panel_root: Path, kind: str) -> dict[str, list[np.ndarray]]:
    values = {name: [] for name in FIELDS}
    for arm in sorted(panel_root.glob(f"arm-*-{kind}")):
        accumulator_path = arm / "replay" / "worker_private_accumulators.npz"
        with np.load(accumulator_path, allow_pickle=False) as source:
            real = np.asarray(source["reduced_real"], dtype=np.float32)
            imag = np.asarray(source["reduced_imag"], dtype=np.float32)
            weight = np.asarray(source["reduced_weight"], dtype=np.float32)
        if real.shape != imag.shape or real.ndim != 2 or real.shape[0] != 2:
            raise RepeatPanelError(f"{arm.name}: replay data half topology differs")
        if weight.shape != real.shape:
            raise RepeatPanelError(f"{arm.name}: replay weight half topology differs")
        values["data_half0"].append(real[0] + np.complex64(1j) * imag[0])
        values["weight_half0"].append(weight[0])
        values["data_half1"].append(real[1] + np.complex64(1j) * imag[1])
        values["weight_half1"].append(weight[1])
    return values


def audit_native_bpref_repeat_panel(
    native_panel_root: Path,
    replay_panel_root: Path,
    *,
    repeat_count: int = 4,
    iteration: int = 1,
) -> dict[str, Any]:
    if repeat_count < 2 or iteration < 1:
        raise RepeatPanelError("native BPref audit requires at least two repeats and iteration >= 1")
    native_panel_root = native_panel_root.resolve()
    replay_panel_root = replay_panel_root.resolve()
    replay_report = audit_replay_repeat_panel(
        replay_panel_root,
        repeat_count=repeat_count,
    )
    native_rows, native_values, native_provenance = _load_native_panel(
        native_panel_root,
        repeat_count=repeat_count,
        iteration=iteration,
    )
    if native_provenance["physical_gpu_uuid"] != replay_report["physical_gpu_uuid"]:
        raise RepeatPanelError("native and replay panels use different physical GPUs")

    native_metrics = {
        name: _pairwise_metrics(native_values[name]) for name in FIELDS
    }
    replay_metrics = {}
    replay_over_native = {}
    for kind in ("private", "shared"):
        replay_values = _load_replay_values(replay_panel_root, kind)
        replay_metrics[kind] = {
            name: _pairwise_metrics(replay_values[name]) for name in FIELDS
        }
        replay_over_native[kind] = {}
        for name in FIELDS:
            native_diameter = native_metrics[name]["maximum_pairwise_relative_l2"]
            replay_diameter = replay_metrics[kind][name][
                "maximum_pairwise_relative_l2"
            ]
            replay_over_native[kind][name] = (
                float(replay_diameter / native_diameter)
                if native_diameter > 0.0
                else None
            )

    return {
        "schema": SCHEMA,
        "result": "complete",
        "scoring": False,
        "scope": (
            "same-physical-GPU pairwise relative-L2 width of native final raw BPref "
            "halves versus exact worker-private and shared host replay"
        ),
        "iteration": iteration,
        "repeat_count": repeat_count,
        "physical_gpu_uuid": native_provenance["physical_gpu_uuid"],
        "native_science_head": native_provenance["science_head"],
        "replay_science_head": replay_report["science_head"],
        "native_arms": native_rows,
        "native_metrics": native_metrics,
        "replay_metrics": replay_metrics,
        "replay_over_native_maximum_diameter": replay_over_native,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-panel-root", required=True, type=Path)
    parser.add_argument("--replay-panel-root", required=True, type=Path)
    parser.add_argument("--repeat-count", type=int, default=4)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit_native_bpref_repeat_panel(
        args.native_panel_root,
        args.replay_panel_root,
        repeat_count=args.repeat_count,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
