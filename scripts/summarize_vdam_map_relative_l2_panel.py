#!/usr/bin/env python3
"""Aggregate candidate VDAM map relative-L2 envelopes into one panel report."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np

INPUT_SCHEMA = "recovar.vdam_map_relative_l2_envelope.v1"
SCHEMA = "recovar.vdam_map_relative_l2_panel.v1"


class MapRelativeL2PanelError(RuntimeError):
    """Raised when candidate envelope reports do not form one sealed panel."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MapRelativeL2PanelError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise MapRelativeL2PanelError(f"cannot read {path}: {exc}") from exc
    _require(isinstance(value, dict), f"report must be an object: {path}")
    return value


def _load_map(path: Path) -> np.ndarray:
    try:
        with mrcfile.open(path, permissive=False) as handle:
            values = np.asarray(handle.data, dtype=np.float64).copy()
    except (OSError, ValueError) as exc:
        raise MapRelativeL2PanelError(f"cannot read map {path}: {exc}") from exc
    _require(values.ndim == 3 and np.all(np.isfinite(values)), f"invalid map: {path}")
    return values


def _symmetric_relative_l2(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    _require(lhs.shape == rhs.shape, f"candidate repeat map shapes differ: {lhs.shape} vs {rhs.shape}")
    denominator = max(float(np.linalg.norm(lhs)), float(np.linalg.norm(rhs)))
    return 0.0 if denominator == 0.0 else float(np.linalg.norm(lhs - rhs) / denominator)


def attach_candidate_repeat_spread(report: dict[str, Any]) -> None:
    """Measure candidate-candidate map spread on the same direct metric."""

    roots = [Path(path) for path in report["candidate_roots"]]
    panel_ratios: list[float] = []
    for row in report["checkpoints"]:
        iteration = int(row["iteration"])
        maps = [
            _load_map(root / "recovar" / f"run_it{iteration:03d}_class001.mrc")
            for root in roots
        ]
        distances = [
            _symmetric_relative_l2(maps[lhs], maps[rhs])
            for lhs, rhs in itertools.combinations(range(len(maps)), 2)
        ]
        maximum = max(distances)
        native_maximum = float(row["native_repeat_max_relative_l2"])
        if native_maximum > 0.0:
            ratio = maximum / native_maximum
        elif maximum == 0.0:
            ratio = 0.0
        else:
            ratio = None
        row["candidate_repeat_relative_l2"] = distances
        row["candidate_repeat_max_relative_l2"] = maximum
        row["candidate_repeat_max_over_native_repeat_max_relative_l2"] = ratio
        if ratio is not None:
            panel_ratios.append(float(ratio))
    report["maximum_candidate_repeat_spread_over_native_repeat_spread"] = (
        max(panel_ratios) if panel_ratios else None
    )


def summarize_map_relative_l2_panel(
    reports: list[dict[str, Any]],
    *,
    expected_candidate_count: int = 4,
) -> dict[str, Any]:
    _require(
        len(reports) == int(expected_candidate_count),
        f"expected {expected_candidate_count} candidate reports, got {len(reports)}",
    )
    _require(expected_candidate_count >= 2, "panel requires at least two candidates")
    for index, report in enumerate(reports, start=1):
        _require(report.get("schema") == INPUT_SCHEMA, f"candidate {index} schema differs")

    first = reports[0]
    shared = {
        "suite_id": first.get("suite_id"),
        "case_id": first.get("case_id"),
        "native_roots": first.get("native_roots"),
        "native_repeat_count": first.get("native_repeat_count"),
        "checkpoint_count": first.get("checkpoint_count"),
    }
    _require(shared["suite_id"] and shared["case_id"], "suite/case identity is missing")
    _require(
        isinstance(shared["native_roots"], list)
        and len(shared["native_roots"]) == int(shared["native_repeat_count"]),
        "native repeat topology is invalid",
    )
    candidate_roots = [str(report.get("candidate_root", "")) for report in reports]
    _require(
        all(candidate_roots) and len(set(candidate_roots)) == len(candidate_roots),
        "candidate roots must be nonempty and unique",
    )
    for index, report in enumerate(reports[1:], start=2):
        for name, expected in shared.items():
            _require(report.get(name) == expected, f"candidate {index} {name} differs")

    checkpoint_lists = [report.get("checkpoints") for report in reports]
    _require(
        all(isinstance(rows, list) for rows in checkpoint_lists),
        "candidate checkpoints must be lists",
    )
    iterations = [int(row["iteration"]) for row in checkpoint_lists[0]]
    _require(
        len(iterations) == int(shared["checkpoint_count"])
        and iterations == sorted(set(iterations)),
        "checkpoint topology is invalid",
    )

    rows: list[dict[str, Any]] = []
    for position, iteration in enumerate(iterations):
        candidate_rows = [checkpoints[position] for checkpoints in checkpoint_lists]
        _require(
            all(int(row["iteration"]) == iteration for row in candidate_rows),
            f"candidate checkpoint topology differs at position {position}",
        )
        native_vectors = [row["native_repeat_relative_l2"] for row in candidate_rows]
        native_maxima = [float(row["native_repeat_max_relative_l2"]) for row in candidate_rows]
        _require(
            all(vector == native_vectors[0] for vector in native_vectors[1:])
            and all(value == native_maxima[0] for value in native_maxima[1:]),
            f"native envelope differs at iteration {iteration}",
        )
        candidate_native = [
            [float(value) for value in row["candidate_native_relative_l2"]]
            for row in candidate_rows
        ]
        _require(
            all(len(values) == int(shared["native_repeat_count"]) for values in candidate_native),
            f"candidate/native matrix differs at iteration {iteration}",
        )
        nearest = [int(np.argmin(values)) + 1 for values in candidate_native]
        within = [
            bool(row["candidate_within_native_repeat_relative_l2_envelope"])
            for row in candidate_rows
        ]
        ratios = [
            row["candidate_best_over_native_repeat_max_relative_l2"]
            for row in candidate_rows
        ]
        finite_ratios = [float(value) for value in ratios if value is not None]
        rows.append(
            {
                "iteration": iteration,
                "candidate_native_relative_l2": candidate_native,
                "candidate_best_native_relative_l2": [
                    float(row["candidate_best_native_relative_l2"])
                    for row in candidate_rows
                ],
                "native_repeat_max_relative_l2": native_maxima[0],
                "candidate_best_over_native_repeat_max_relative_l2": ratios,
                "maximum_finite_candidate_over_native_repeat_envelope_ratio": (
                    max(finite_ratios) if finite_ratios else None
                ),
                "candidate_within_native_repeat_relative_l2_envelope": within,
                "all_candidates_within_native_repeat_relative_l2_envelope": all(within),
                "nearest_native_repeat_by_candidate": nearest,
                "nearest_native_repeat_coverage": sorted(set(nearest)),
            }
        )

    outside = [row for row in rows if not row["all_candidates_within_native_repeat_relative_l2_envelope"]]
    panel_ratios = [
        row["maximum_finite_candidate_over_native_repeat_envelope_ratio"]
        for row in rows
        if row["maximum_finite_candidate_over_native_repeat_envelope_ratio"] is not None
    ]
    return {
        "schema": SCHEMA,
        "status": "pass" if not outside else "fail",
        "classification": (
            "all_candidate_maps_inside_native_repeat_direct_relative_l2_envelope"
            if not outside
            else "candidate_map_outside_native_repeat_direct_relative_l2_envelope"
        ),
        "scope": "direct scale-sensitive map distribution diagnostic only; no acceptance effect",
        "suite_id": shared["suite_id"],
        "case_id": shared["case_id"],
        "candidate_roots": candidate_roots,
        "native_roots": shared["native_roots"],
        "candidate_count": len(reports),
        "native_repeat_count": shared["native_repeat_count"],
        "checkpoint_count": len(rows),
        "first_iteration_outside_native_repeat_envelope": (
            None if not outside else int(outside[0]["iteration"])
        ),
        "outside_candidate_checkpoint_count": sum(
            sum(not flag for flag in row["candidate_within_native_repeat_relative_l2_envelope"])
            for row in rows
        ),
        "maximum_finite_candidate_over_native_repeat_envelope_ratio": (
            max(panel_ratios) if panel_ratios else None
        ),
        "checkpoints": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--expected-candidate-count", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    paths = [path.resolve() for path in args.report]
    report = summarize_map_relative_l2_panel(
        [_load(path) for path in paths],
        expected_candidate_count=args.expected_candidate_count,
    )
    attach_candidate_repeat_spread(report)
    report["provenance"] = {
        "input_report_sha256": {str(path): _sha256(path) for path in paths}
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
