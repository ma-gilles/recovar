#!/usr/bin/env python3
"""Audit native-repeat VDAM operands replayed through exact BPref kernels."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audit_vdam_native_bpref_repeat_panel import (
    audit_native_bpref_repeat_panel,
)
from scripts.audit_vdam_repeat_panel import RepeatPanelError, _load_json
from scripts.run_vdam_worker_private_host_replay import (
    _load_native_panels,
    _load_native_topology,
)

SCHEMA = "recovar.vdam_native_operand_replay_panel.v1"


def _normalized_weights(panel: dict[str, object]) -> np.ndarray:
    raw = np.asarray(panel["weights"], dtype=np.float64)
    positive = raw > 0.0
    if not np.any(positive):
        raise RepeatPanelError("native panel has no positive posterior weights")
    result = np.where(positive, raw, 0.0)
    result /= np.sum(result, dtype=np.float64)
    return result


def _pairwise_width(values: list[np.ndarray]) -> dict[str, Any]:
    rows = []
    for (left_index, left), (right_index, right) in itertools.combinations(
        enumerate(values, start=1), 2
    ):
        denominator = float(np.linalg.norm(left.ravel()))
        if denominator == 0.0 or not np.isfinite(denominator):
            raise RepeatPanelError("native posterior has zero or nonfinite norm")
        rows.append(
            {
                "left_repeat": left_index,
                "right_repeat": right_index,
                "relative_l2": float(
                    np.linalg.norm((right - left).ravel()) / denominator
                ),
                "max_absolute_error": float(np.max(np.abs(right - left))),
            }
        )
    return {
        "minimum_pairwise_relative_l2": min(row["relative_l2"] for row in rows),
        "maximum_pairwise_relative_l2": max(row["relative_l2"] for row in rows),
        "pairwise": rows,
    }


def _expected_repeat_paths(panel_root: Path, repeat: int) -> dict[str, Path]:
    root = panel_root / f"repeat-{repeat:02d}"
    return {
        "native_topology": (root / "provenance" / "bpref_topology.tsv").resolve(),
        "native_topology_data_star": (root / "relion" / "run_it001_data.star").resolve(),
        "native_panel_directory": (root / "native_panels").resolve(),
    }


def _validate_replay_paths(panel_root: Path, repeat_count: int) -> None:
    for kind in ("private", "shared"):
        arms = sorted(panel_root.glob(f"arm-*-{kind}"))
        if len(arms) != repeat_count:
            raise RepeatPanelError(
                f"expected {repeat_count} {kind} native-operand arms, found {len(arms)}"
            )
        for repeat, arm in enumerate(arms, start=1):
            provenance = _load_json(arm / "provenance.json", label=f"{arm.name} provenance")
            report = _load_json(
                arm / "replay" / "worker_private_report.json",
                label=f"{arm.name} report",
            )
            if provenance.get("native_panel_weights") is not True:
                raise RepeatPanelError(
                    f"{arm.name}: provenance did not enable native panel weights"
                )
            if report.get("native_panel_weights_replayed") is not True:
                raise RepeatPanelError(
                    f"{arm.name}: report did not replay native panel weights"
                )
            for name, expected in _expected_repeat_paths(panel_root, repeat).items():
                actual = Path(str(provenance.get(name, ""))).resolve()
                if actual != expected:
                    raise RepeatPanelError(
                        f"{arm.name}: {name} does not come from native repeat {repeat}"
                    )


def _native_operand_metrics(panel_root: Path, repeat_count: int) -> dict[str, Any]:
    panels_by_repeat = []
    topology_by_repeat = []
    for repeat in range(1, repeat_count + 1):
        paths = _expected_repeat_paths(panel_root, repeat)
        panels_by_repeat.append(
            _load_native_panels(
                paths["native_panel_directory"],
                paths["native_topology_data_star"],
                iteration=1,
            )
        )
        topology_by_repeat.append(
            _load_native_topology(
                paths["native_topology"],
                paths["native_topology_data_star"],
                iteration=1,
            )
        )
    identities = set(panels_by_repeat[0])
    if not identities or any(set(panel) != identities for panel in panels_by_repeat[1:]):
        raise RepeatPanelError("native repeat operand identities differ")
    if any(set(topology) != identities for topology in topology_by_repeat):
        raise RepeatPanelError("native repeat topology identities differ from panels")

    pooled = [[] for _ in range(repeat_count)]
    support_mismatch_count = 0
    euler_mismatch_count = 0
    owner_mismatch_count = 0
    launch_count_mismatch_count = 0
    particle_widths = []
    for identity in sorted(identities):
        repeat_panels = [panels[identity] for panels in panels_by_repeat]
        weights = [_normalized_weights(panel) for panel in repeat_panels]
        if len({value.shape for value in weights}) != 1:
            raise RepeatPanelError(f"particle {identity}: native posterior shapes differ")
        supports = [value > 0.0 for value in weights]
        support_mismatch_count += int(
            np.count_nonzero(np.logical_or.reduce(supports) != np.logical_and.reduce(supports))
        )
        reference_eulers = np.asarray(repeat_panels[0]["eulers"], dtype=np.float32)
        euler_mismatch_count += sum(
            int(not np.array_equal(reference_eulers, np.asarray(panel["eulers"], dtype=np.float32)))
            for panel in repeat_panels[1:]
        )
        owners_counts = [topology[identity] for topology in topology_by_repeat]
        owner_mismatch_count += sum(
            int(value[0] != owners_counts[0][0]) for value in owners_counts[1:]
        )
        launch_count_mismatch_count += sum(
            int(value[1] != owners_counts[0][1]) for value in owners_counts[1:]
        )
        for repeat, value in enumerate(weights):
            pooled[repeat].append(value.ravel())
        particle_widths.append(_pairwise_width(weights)["maximum_pairwise_relative_l2"])

    pooled_values = [np.concatenate(values) for values in pooled]
    return {
        "particle_count": len(identities),
        "pooled_normalized_posterior": _pairwise_width(pooled_values),
        "particle_maximum_pairwise_relative_l2": {
            "minimum": float(np.min(particle_widths)),
            "median": float(np.median(particle_widths)),
            "maximum": float(np.max(particle_widths)),
        },
        "support_mismatch_coordinate_count": support_mismatch_count,
        "euler_repeat_mismatch_count": euler_mismatch_count,
        "worker_owner_repeat_mismatch_count": owner_mismatch_count,
        "launch_count_repeat_mismatch_count": launch_count_mismatch_count,
    }


def audit_native_operand_replay_panel(
    panel_root: Path,
    *,
    repeat_count: int = 4,
    iteration: int = 1,
) -> dict[str, Any]:
    panel_root = panel_root.resolve()
    submission = _load_json(
        panel_root / "submission_provenance.json", label="submission provenance"
    )
    if submission.get("native_repeat_operands") is not True:
        raise RepeatPanelError("submission did not replay each native repeat's operands")
    _validate_replay_paths(panel_root, repeat_count)
    report = audit_native_bpref_repeat_panel(
        panel_root,
        panel_root,
        repeat_count=repeat_count,
        iteration=iteration,
    )
    report["schema"] = SCHEMA
    report["scope"] = (
        "same-allocation native raw BPref width versus exact private/shared replay "
        "using each native repeat's own owners, grid counts, Euler rows, and posteriors"
    )
    report["native_repeat_operands"] = _native_operand_metrics(
        panel_root, repeat_count
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-panel-root", required=True, type=Path)
    parser.add_argument("--replay-panel-root", required=True, type=Path)
    parser.add_argument("--repeat-count", type=int, default=4)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.native_panel_root.resolve() != args.replay_panel_root.resolve():
        raise RepeatPanelError("native-operand replay requires one matched panel root")
    report = audit_native_operand_replay_panel(
        args.native_panel_root,
        repeat_count=args.repeat_count,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
