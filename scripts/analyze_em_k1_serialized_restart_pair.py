#!/usr/bin/env python3
"""Classify the fixed Case-22 iteration-0/iteration-1 restart pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.analyze_em_k1_serialized_restart_boundary import (
    CLASSIFICATION as SCORE_CLASSIFICATION,
)
from scripts.analyze_em_k1_serialized_restart_maps import (
    CLASSIFICATION as MAP_CLASSIFICATION,
)

SCHEMA = "em-k1-case22-serialized-restart-pair-v1"
PAIR_CLASSIFICATIONS = {
    (True, True): "both_serialized_restart_points_close_score_and_map_gates",
    (True, False): "only_iteration0_restart_closes_score_and_map_gates",
    (False, True): "only_iteration1_restart_closes_score_and_map_gates",
    (False, False): "neither_serialized_restart_point_closes_score_and_map_gates",
}
CAUSAL_INTERPRETATIONS = {
    (True, True): (
        "case22_iteration2_gap_depends_on_process_resident_state_not_"
        "preserved_by_either_serialized_restart"
    ),
    (True, False): (
        "case22_recovery_requires_replaying_iteration1_from_serialized_it0"
    ),
    (False, True): "case22_recovery_is_direct_iteration1_restart_specific",
    (False, False): "serialized_restart_hypothesis_rejected_for_case22",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_restart_pair(
    *,
    iteration0_accepted: bool,
    iteration1_accepted: bool,
) -> tuple[str, str]:
    key = (bool(iteration0_accepted), bool(iteration1_accepted))
    return PAIR_CLASSIFICATIONS[key], CAUSAL_INTERPRETATIONS[key]


def _load_score(path: Path) -> tuple[dict[str, Any], bool]:
    report = json.loads(path.read_text())
    _require(
        report.get("schema") == "em-k1-serialized-restart-boundary-v1",
        "serialized-restart score schema changed",
    )
    _require(
        report.get("status") == "complete"
        and report.get("classification_ready") is True,
        "serialized-restart score report is incomplete",
    )
    fixed = report.get("fixed_metric", {})
    _require(
        fixed.get("evaluated_particles") == 14
        and fixed.get("expected_particles") == 14,
        "serialized-restart score denominator changed",
    )
    passed = bool(
        report.get("classification") == SCORE_CLASSIFICATION
        and fixed.get("serialized_restart_dominated") == 14
        and fixed.get("absolute_score_gate_passed") == 14
    )
    return report, passed


def _load_map(
    path: Path,
    *,
    expected_score_path: Path,
    expected_score_passed: bool,
) -> tuple[dict[str, Any], bool]:
    report = json.loads(path.read_text())
    _require(
        report.get("schema") == "em-k1-serialized-restart-map-fsc-v2",
        "serialized-restart map schema changed",
    )
    _require(
        report.get("status") == "complete"
        and report.get("classification_ready") is True,
        "serialized-restart map report is incomplete",
    )
    fixed = report.get("fixed_metric", {})
    _require(
        fixed.get("evaluated_maps") == 3
        and fixed.get("expected_maps") == 3
        and fixed.get("score_boundary_passed") is expected_score_passed,
        "serialized-restart map denominator or score linkage changed",
    )
    score_link = report.get("score_boundary", {})
    _require(
        Path(score_link.get("path", "")).resolve()
        == expected_score_path.resolve()
        and score_link.get("sha256") == _sha256(expected_score_path)
        and score_link.get("passed") is expected_score_passed,
        "serialized-restart map report is not hash-linked to its score report",
    )
    passed = bool(
        expected_score_passed
        and report.get("classification") == MAP_CLASSIFICATION
        and fixed.get("parity_strictly_improved") == 3
        and fixed.get("gt_nondegraded") == 3
        and report.get("overall_intervention_accepted") is True
    )
    _require(
        report.get("overall_intervention_accepted") is passed,
        "serialized-restart overall acceptance does not replay",
    )
    return report, passed


def build_report(
    *,
    iteration0_score_path: Path,
    iteration0_map_path: Path,
    iteration1_score_path: Path,
    iteration1_map_path: Path,
) -> dict[str, Any]:
    arms = {}
    accepted = {}
    for label, score_path, map_path in (
        ("iteration0_restart", iteration0_score_path, iteration0_map_path),
        ("iteration1_restart", iteration1_score_path, iteration1_map_path),
    ):
        score, score_passed = _load_score(score_path)
        maps, arm_accepted = _load_map(
            map_path,
            expected_score_path=score_path,
            expected_score_passed=score_passed,
        )
        accepted[label] = arm_accepted
        arms[label] = {
            "score_passed": score_passed,
            "map_classification": maps["classification"],
            "map_fixed_metric": maps["fixed_metric"],
            "overall_accepted": arm_accepted,
            "score": {
                "path": str(score_path.resolve()),
                "sha256": _sha256(score_path),
                "classification": score["classification"],
                "fixed_metric": score["fixed_metric"],
            },
            "map": {
                "path": str(map_path.resolve()),
                "sha256": _sha256(map_path),
            },
        }

    classification, causal_interpretation = classify_restart_pair(
        iteration0_accepted=accepted["iteration0_restart"],
        iteration1_accepted=accepted["iteration1_restart"],
    )
    score_pass_count = sum(
        int(arm["score_passed"]) for arm in arms.values()
    )
    map_parity_pass_count = sum(
        int(arm["map_fixed_metric"]["parity_strictly_improved"] == 3)
        for arm in arms.values()
    )
    map_gt_pass_count = sum(
        int(arm["map_fixed_metric"]["gt_nondegraded"] == 3)
        for arm in arms.values()
    )
    overall_pass_count = sum(
        int(arm["overall_accepted"]) for arm in arms.values()
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification_ready": True,
        "classification": classification,
        "causal_interpretation": causal_interpretation,
        "scorecard_change_admissible": False,
        "metric_policy": (
            "fixed two-restart-arm denominator; each arm requires 14/14 "
            "score gates, 3/3 strictly positive parity FSC-AUC deltas, and "
            "3/3 nonnegative GT FSC-AUC deltas; no fitted tolerance; "
            "no correlation"
        ),
        "fixed_metric": {
            "evaluated_restart_arms": 2,
            "expected_restart_arms": 2,
            "score_gate_passed_arms": score_pass_count,
            "map_parity_gate_passed_arms": map_parity_pass_count,
            "map_gt_gate_passed_arms": map_gt_pass_count,
            "overall_passed_arms": overall_pass_count,
        },
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration0-score", type=Path, required=True)
    parser.add_argument("--iteration0-map", type=Path, required=True)
    parser.add_argument("--iteration1-score", type=Path, required=True)
    parser.add_argument("--iteration1-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    report = build_report(
        iteration0_score_path=args.iteration0_score,
        iteration0_map_path=args.iteration0_map,
        iteration1_score_path=args.iteration1_score,
        iteration1_map_path=args.iteration1_map,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "causal_interpretation": report["causal_interpretation"],
                "fixed_metric": report["fixed_metric"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
