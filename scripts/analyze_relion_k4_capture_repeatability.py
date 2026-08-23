#!/usr/bin/env python3
"""Measure run-to-run repeatability of a fixed K=4 RELION capture panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ACTIVE = np.uint32(8)
PANEL_SCHEMA = "recovar.k4_iter10_class2_residual_target_panel.v1"
REPORT_SCHEMA = "relion.k4_iter10_panel12_capture_repeatability.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _center(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    _require(np.all(np.isfinite(array)), "capture contains non-finite scores")
    return array - np.mean(array, dtype=np.float64) if array.size else array


def _residual(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float | int | bool]:
    left = np.asarray(lhs, dtype=np.float64)
    right = np.asarray(rhs, dtype=np.float64)
    _require(left.shape == right.shape, "capture score shape changed")
    delta = right - left
    energy = float(np.vdot(delta, delta).real)
    return {
        "candidate_count": int(delta.size),
        "exact_equal": bool(np.array_equal(left, right)),
        "mismatch_count": int(np.count_nonzero(left != right)),
        "residual_l2": float(np.sqrt(energy)),
        "residual_energy": energy,
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "median_abs": float(np.median(np.abs(delta))) if delta.size else 0.0,
        "p95_abs": float(np.quantile(np.abs(delta), 0.95)) if delta.size else 0.0,
    }


def _load_capture_directory(
    directory: Path,
    expected_stacks: set[int],
    load_factor_capture,
    load_fine_score_capture,
) -> tuple[dict[int, object], dict[int, object]]:
    factor_paths = sorted(directory.glob("*.bpre-v2.bin"))
    score_paths = sorted(directory.glob("*.fine-score-v1.bin"))
    _require(len(factor_paths) == len(expected_stacks), "BPref capture count changed")
    _require(len(score_paths) == len(expected_stacks), "fine-score capture count changed")
    factors = {capture.stack_index: capture for capture in map(load_factor_capture, factor_paths)}
    scores = {capture.stack_index: capture for capture in map(load_fine_score_capture, score_paths)}
    _require(set(factors) == expected_stacks, "BPref stack set changed")
    _require(set(scores) == expected_stacks, "fine-score stack set changed")
    return factors, scores


def analyze(
    *,
    repo: Path,
    panel_json: Path,
    capture_a: Path,
    capture_b: Path,
) -> dict[str, object]:
    sys.path.insert(0, str(repo / "scripts"))
    from validate_relion_bpref_factor_capture import load_factor_capture
    from validate_relion_fine_score_capture import load_fine_score_capture

    panel = json.loads(panel_json.read_text())
    _require(panel.get("schema") == PANEL_SCHEMA, "panel schema changed")
    targets = panel.get("targets")
    _require(isinstance(targets, list) and len(targets) == 12, "panel target count changed")
    expected_stacks = {int(target["zero_based_identity_row"]) + 1 for target in targets}
    _require(len(expected_stacks) == len(targets), "panel contains duplicate stack identities")

    factors_a, scores_a = _load_capture_directory(
        capture_a,
        expected_stacks,
        load_factor_capture,
        load_fine_score_capture,
    )
    factors_b, scores_b = _load_capture_directory(
        capture_b,
        expected_stacks,
        load_factor_capture,
        load_fine_score_capture,
    )

    aggregate_data_a: list[np.ndarray] = []
    aggregate_data_b: list[np.ndarray] = []
    aggregate_combined_a: list[np.ndarray] = []
    aggregate_combined_b: list[np.ndarray] = []
    rows = []
    topology_fields = (
        "sparse_index",
        "rotation_id",
        "rotation_local",
        "translation_id",
        "coarse_translation",
        "flags",
    )
    for target in targets:
        stack_index = int(target["zero_based_identity_row"]) + 1
        factor_a = factors_a[stack_index]
        factor_b = factors_b[stack_index]
        score_a = scores_a[stack_index]
        score_b = scores_b[stack_index]
        _require(factor_a.geometry_only and factor_b.geometry_only, "factor is not geometry-only")
        _require(
            score_a.candidates.shape == score_b.candidates.shape,
            f"stack {stack_index}: candidate shape changed",
        )
        topology_exact = {
            field: bool(np.array_equal(score_a.candidates[field], score_b.candidates[field]))
            for field in topology_fields
        }
        _require(all(topology_exact.values()), f"stack {stack_index}: captured candidate topology changed")
        active_a = (score_a.candidates["flags"] & ACTIVE) != 0
        active_b = (score_b.candidates["flags"] & ACTIVE) != 0
        _require(np.array_equal(active_a, active_b), f"stack {stack_index}: active support changed")

        raw_a = _center(score_a.candidates["raw_diff2"][active_a])
        raw_b = _center(score_b.candidates["raw_diff2"][active_b])
        combined_a = _center(score_a.candidates["combined_preexponent"][active_a])
        combined_b = _center(score_b.candidates["combined_preexponent"][active_b])
        aggregate_data_a.append(raw_a)
        aggregate_data_b.append(raw_b)
        aggregate_combined_a.append(combined_a)
        aggregate_combined_b.append(combined_b)
        winner_a = int(np.argmax(combined_a)) if combined_a.size else None
        winner_b = int(np.argmax(combined_b)) if combined_b.size else None
        rows.append(
            {
                **target,
                "stack_index_one_based": stack_index,
                "active_candidate_count": int(np.count_nonzero(active_a)),
                "candidate_topology_exact": topology_exact,
                "factor_rotations_exact": bool(np.array_equal(factor_a.rotations, factor_b.rotations)),
                "factor_translations_exact": bool(
                    np.array_equal(factor_a.translations, factor_b.translations)
                ),
                "factor_header_differing_indices": [
                    index
                    for index, (lhs, rhs) in enumerate(zip(factor_a.header, factor_b.header, strict=True))
                    if lhs != rhs
                ],
                "fine_score_header_differing_indices": [
                    index
                    for index, (lhs, rhs) in enumerate(zip(score_a.header, score_b.header, strict=True))
                    if lhs != rhs
                ],
                "centered_raw_diff2_repeatability": _residual(raw_a, raw_b),
                "centered_combined_repeatability": _residual(combined_a, combined_b),
                "winner": {
                    "defined": winner_a is not None,
                    "capture_a_flat": winner_a,
                    "capture_b_flat": winner_b,
                    "exact": winner_a == winner_b,
                },
                "artifacts": {
                    "factor_a": {"path": str(factor_a.path), "sha256": factor_a.sha256},
                    "factor_b": {"path": str(factor_b.path), "sha256": factor_b.sha256},
                    "fine_score_a": {"path": str(score_a.path), "sha256": score_a.sha256},
                    "fine_score_b": {"path": str(score_b.path), "sha256": score_b.sha256},
                },
            }
        )

    aggregate = {
        "centered_raw_diff2": _residual(
            np.concatenate(aggregate_data_a),
            np.concatenate(aggregate_data_b),
        ),
        "centered_combined": _residual(
            np.concatenate(aggregate_combined_a),
            np.concatenate(aggregate_combined_b),
        ),
    }
    geometry_exact = all(
        row["factor_rotations_exact"] and row["factor_translations_exact"] for row in rows
    )
    winners_exact = all(row["winner"]["exact"] for row in rows)
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "classification": "capture_repeatability_floor_measured",
        "scorecard_change_admissible": False,
        "scope": {
            "physical_iteration": 10,
            "class_one_based": 2,
            "target_count": len(rows),
            "geometry_exact_all": geometry_exact,
            "winner_exact_count": sum(row["winner"]["exact"] for row in rows),
            "winner_evaluable_count": sum(row["winner"]["defined"] for row in rows),
            "winners_exact_all": winners_exact,
        },
        "aggregate": aggregate,
        "targets": rows,
        "inputs": {
            "panel": {"path": str(panel_json), "sha256": _sha256(panel_json)},
            "capture_a": str(capture_a),
            "capture_b": str(capture_b),
            "analyzer_repo_head": "",
        },
        "next_step": (
            "Compare the host-vs-RELION-CUDA residual change against this capture-vs-capture "
            "uncertainty floor; do not promote the fixed scorecard from this diagnostic."
        ),
    }


def _clean_repo_head(repo: Path) -> str:
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain=v1"], text=True)
    _require(not status, "analyzer repository is dirty")
    return head


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--panel-json", required=True, type=Path)
    parser.add_argument("--capture-a", required=True, type=Path)
    parser.add_argument("--capture-b", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        repo=args.repo,
        panel_json=args.panel_json,
        capture_a=args.capture_a,
        capture_b=args.capture_b,
    )
    report["inputs"]["analyzer_repo_head"] = _clean_repo_head(args.repo)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"classification": report["classification"], **report["scope"]}, indent=2))


if __name__ == "__main__":
    main()
